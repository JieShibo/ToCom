# ToCom
Official code of ''Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning''

<p align="left">
<a href="https://arxiv.org/abs/2408.06798" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2408.06798-b31b1b.svg?style=flat" /></a>
</p>

## Requirements
+ pytorch >= 1.12.1 
+ torchvision
+ timm==0.4.12
+ [ToMe](https://github.com/facebookresearch/ToMe)

## Usage

### Datasets
  - [ImageNet](https://www.image-net.org/index.php)

  - [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

  - [CUB200 2011](https://data.caltech.edu/records/65de6-vp158)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

### Pre-training DeiT
Download the pre-trained DeiT-B model from [here](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth).

### Pre-training ToCom
```bash
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_base_patch16_224_tocom \
    --batch-size 128 \
    --data-path <imagenet-path> \
    --output_dir ./ckpt/tocom_tome \
    --epochs 10 \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 0 \
    --distillation-type soft \
    --distillation-alpha 1.0 \
    --teacher-model deit_base_patch16_224 \
    --teacher-path ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --lr 1e-3 \
    --r_src -1
```

### Get Downstream Off-the-shelf Checkpoint
```bash
# CIFAR-100
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 128 \
    --data-path <cifar-path> \
    --output_dir ./ckpt/downstream/CIFAR \
    --epochs 100 \
    --data-set CIFAR \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 10 \
    --lr 5e-5 \
    --r_src <source_r>

# CUB-200
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 64 \
    --data-path <cub-path> \
    --output_dir ./ckpt/downstream/CUB \
    --epochs 30 \
    --data-set CUB \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 5 \
    --lr 1e-3 \
    --r_src <source_r>

# CARS
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 64 \
    --data-path <cars-path> \
    --output_dir ./ckpt/downstream/CARS \
    --epochs 30 \
    --data-set CARS \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 5 \
    --lr 2e-3 \
    --r_src <source_r>

# DOGS
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 64 \
    --data-path <dogs-path> \
    --output_dir ./ckpt/downstream/DOGS \
    --epochs 30 \
    --data-set DOGS \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 5 \
    --lr 2e-4 \
    --r_src <source_r>

# FLOWERS
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224 \
    --batch-size 64 \
    --data-path <flowers-path> \
    --output_dir ./ckpt/downstream/FLOWERS \
    --epochs 30 \
    --data-set FLOWERS \
    --finetune ./ckpt/deit_base_patch16_224-b5f2ef4d.pth \
    --warmup-epochs 5 \
    --lr 2e-3 \
    --r_src <source_r>
```

### Evaluation
```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
    --model deit_base_patch16_224_tocom \
    --batch-size 128 \
    --data-path <dataset-path> \
    --data-set <dataset> \ # CIFAR, CUB, CARS, DOGS, FLOWERS
    --finetune ./ckpt/downstream/<dataset>/best_checkpoint.pth \
    --eval \
    --r_src <source_r> \ # same as training
    --r_tgt <target_r> \ # from 0 to 16
    --tocom_scale <tocom_scale> \ # search in [0.05, 0.08, 0.1, 0.12, 0.15], set to 0 to disable ToCom
    --tocom_path ./ckpt/tocom_tome/checkpoint.pth
```

### Checkpoints
To be uploaded.

## Acknowledgement
This code is built on [ToMe](https://github.com/facebookresearch/ToMe) and [DeiT](https://github.com/facebookresearch/deit).

## Citation
```
@inproceedings{tocom,
  author       = {Shibo Jie and Yehui Tang and Jianyuan Guo and Zhi-Hong Deng and Kai Han and Yunhe Wang},
  title        = {Token Compensator: Altering Inference Cost of Vision Transformer without Re-Tuning},
  booktitle    = {Proceedings of ECCV},
  year         = {2024}
}
```