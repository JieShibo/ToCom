from __future__ import print_function
from PIL import Image
from os.path import join
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir, list_files

class StanfordDogs(data.Dataset):
    folder = 'StanfordDogs'
    download_url_prefix = 'http://vision.stanford.edu/aditya86/ImageNetDogs/'

    def __init__(self, root, train=True, cropped=False, transform=None, target_transform=None, download=False):
        self.root = join(os.path.expanduser(root), self.folder)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.cropped = cropped

        if download:
            self.download()
    
        split = self.load_split()
        self.images_folder = join(self.root, 'Images')
        self.annotations_folder = join(self.root, 'Annotation')
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [[(annotation, box, idx) for box in self.get_boxes(join(self.annotations_folder, annotation))] 
                                       for annotation, idx in split]
            self._flat_breed_annotations = sum(self._breed_annotations, [])
            self._flat_bread_images = [(annotation+'.jpg', idx) for annotation, box, idx in self._flat_breed_annotations]
        else:
            self._breed_annotations = [(annotation+'.jpg', idx) for annotation, idx in split]
            self._flat_breed_annotations = self._breed_annotations

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        image_path, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_path)
        image = Image.open(image_path).convert('RGB')
        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target_class = self.target_transform(target_class)
        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, 'Images')) and os.path.exists(join(self.root, 'Annotation')):
            if len(os.listdir(join(self.root, 'Images'))) == len(os.listdir(join(self.root, 'Annotation'))) == 120:
                print('Files already downloaded and verified')
                return
        
        for filename in ['images', 'annotation', 'lists']:
            tar_filename = filename + '.tar'
            url = self.download_url_prefix + '/' + tar_filename
            download_url(url, self.root, tar_filename, None)
            print('Extracting downloaded files: ' + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), 'r') as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    @staticmethod
    def get_boxes(path):
        import xml.etree.ElementTree
        e = xml.etree.ElementTree.parse(path).getroot()
        boxes = []
        for objs in e.iter('object'):
            boxes.append([int(objs.find('bndbox').find('xmin').text),
                           int(objs.find('bndbox').find('ymin').text),
                           int(objs.find('bndbox').find('xmax').text),
                           int(objs.find('bndbox').find('ymax').text)])
        return boxes    

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, 'train_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'train_list.mat'))['labels']
        else:
            split = scipy.io.loadmat(join(self.root, 'test_list.mat'))['annotation_list']
            labels = scipy.io.loadmat(join(self.root, 'test_list.mat'))['labels']
        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))

 

        
        