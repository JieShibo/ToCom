# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import math
import tome 
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r


__all__ = [
    'deit_base_patch16_224',
    'deit_base_patch16_224_tocom',
]

class Adapter(nn.Module):
    def __init__(self, dim, act=True):
        super().__init__()    
        assert dim % 16 == 0
        self.adapter_down = nn.Linear(768, dim, bias=False)
        self.adapter_up = nn.Linear(dim, 768, bias=False)
        nn.init.zeros_(self.adapter_up.weight)
        if act:
            self.act = nn.GELU()
        else:
            self.act = nn.Identity()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
    def forward(self, x, r_tgt=0, r_src=0):
        B, N, C = x.shape
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        mask_src = torch.tensor([0.] * (self.dim - self.dim // 16 * r_src) + [1.] * (self.dim // 16)  * r_src).float().to(x.device).reshape(1, 1, self.dim)
        mask_tgt = torch.tensor([0.] * (self.dim - self.dim // 16  * r_tgt) + [1.] * (self.dim // 16)  * r_tgt).float().to(x.device).reshape(1, 1, self.dim)
        x_down = x_down * (mask_tgt - mask_src)
        x_up = self.adapter_up(x_down)
        return x_up


def forward_block_tome(self, x):
    r_tgt, r_src = self._tome_info["r"].pop(0)
    attn_size = self._tome_info["size"] if self._tome_info["prop_attn"] else None
    x_attn, metric = self.attn(self.norm1(x), attn_size, r_tgt, r_src)
    x = x + self._drop_path1(x_attn)
    if r_tgt > 0:
        merge, _ = bipartite_soft_matching(metric, r_tgt, self._tome_info["class_token"], self._tome_info["distill_token"], )
        if self._tome_info["trace_source"]:
            self._tome_info["source"] = merge_source(merge, x, self._tome_info["source"])
        x, self._tome_info["size"] = merge_wavg(merge, x, self._tome_info["size"])
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def forward_attn(self, x, size, r_tgt, r_src):
    B, N, C = x.shape
    qkv = (
        self.qkv(x)
        .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    if r_tgt != r_src:
        q_delta = self.q_tocom(x, r_tgt, r_src).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.s
        v_delta = self.v_tocom(x, r_tgt, r_src).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) * self.s
    else:
        q_delta = v_delta = 0
    q, k, v = (
        qkv[0] + q_delta,
        qkv[1],
        qkv[2] + v_delta
    )
    attn = (q @ k.transpose(-2, -1)) * self.scale

    if size is not None:
        attn = attn + size.log()[:, None, None, :, 0]
    
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)
    x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x, k.mean(dim=1)

def set_lora(model, dim=32, s=1):
    for layer in model.children():
        if isinstance(layer, timm.models.vision_transformer.Block):
            layer.attn.q_tocom = Adapter(dim, act=False)
            layer.attn.v_tocom = Adapter(dim, act=False)
            layer.attn.s = s
            bound_method = forward_attn.__get__(layer.attn, layer.attn.__class__)
            setattr(layer.attn, "forward", bound_method)
            bound_method = forward_block_tome.__get__(layer, layer.__class__)
            setattr(layer, "forward", bound_method)
        elif len(list(layer.children())) > 0:
            set_lora(layer, dim, s)


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    if 'tocom_scale' in kwargs:
        del kwargs['tocom_scale']
    if 'tocom_dim' in kwargs:
        del kwargs['tocom_dim']
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], False)
    import tome
    tome.patch.timm(model, prop_attn=True)
    return model


@register_model
def deit_base_patch16_224_tocom(pretrained=False, **kwargs):
    tocom_scale = kwargs['tocom_scale']
    tocom_dim = kwargs['tocom_dim']
    del kwargs['tocom_scale']
    del kwargs['tocom_dim']
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"], False)
    import tome
    tome.patch.timm(model, prop_attn=True)
    set_lora(model, tocom_dim, tocom_scale)
    for n, p in model.named_parameters():
        if 'adapter' in n:
            p.requires_grad = True
        else:
            p.requires_grad = False
    return model





