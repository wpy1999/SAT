import torch
import os
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import torch.utils.data as data
from torchvision.datasets import ImageFolder

def bbox_crop(bbox, resize_size, crop_size):
    shift_size = (resize_size - crop_size) // 2.
    x0, y0, x1, y1, h, w = bbox.split(' ')
    x0, y0, x1, y1, h, w = float(x0), float(y0), float(x1), float(y1), float(h), float(w)
    x0 = int(max(x0 / w * resize_size - shift_size, 0))
    y0 = int(max(y0 / h * resize_size - shift_size, 0))
    x1 = int(min(x1 / w * resize_size - shift_size, crop_size - 1))
    y1 = int(min(y1 / h * resize_size - shift_size, crop_size - 1))
    return np.array([x0, y0, x1, y1]).reshape(-1)

def load_test_bbox(image_list, label_classes, phase):
    img_dict = []
    with open(image_list, 'r') as f:
        for line in f:
            img_name, classes, img_type, bbox = line[:-1].split(';')
            if (img_type == '0' or img_type == '2') and phase == 'train':
                img_dict.append(img_name + ';' + str(label_classes[classes]))
            elif img_type == '1' and phase == 'test': ## test
                img_dict.append(img_name + ';' + str(label_classes[classes]) + ';' + bbox)
    return img_dict
    
class ImageDataset(data.Dataset):
    def __init__(self, args, phase=None):
        args.num_classes = 100
        self.args =args
        self.root = 'Data/' + args.root
        self.image_list = self.root + '/' + 'images_box.txt'
        self.classes_list = self.root + '/' + 'classes.txt'
        self.crop_size = args.crop_size 
        self.resize_size = args.resize_size
        self.phase = args.phase if phase == None else phase
        self.num_classes = args.num_classes
        if self.phase == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        elif self.phase == 'test':
            self.transform = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        self.label_classes = {} 
        with open(self.classes_list, 'r') as f:
            for i, line in enumerate(f):
                self.label_classes[line[:-1]] = i
        self.index_list = load_test_bbox(self.image_list, self.label_classes, self.phase)

    def __getitem__(self, index):
        if self.phase == 'train':
            img_name, img_class = self.index_list[index].split(';')
            path = self.root + '/images/'  + img_name + '.jpg'
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            img_class = int(img_class)
            return path, img, img_class
        else:
            img_name, img_class, bbox = self.index_list[index].split(';')
            path = self.root + '/images/' + img_name  + '.jpg'
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            bbox = bbox_crop(bbox, self.resize_size, self.crop_size)
            img_class = int(img_class)
            return img, img_class, bbox, path

    def __len__(self):
        return len(self.index_list)