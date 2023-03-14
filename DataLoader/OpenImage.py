import torch
import os
import torch.nn as nn 
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
import torch.utils.data as data
from torchvision.datasets import ImageFolder

def load_img_list(image_list_path):
    img_dict = []
    with open(image_list_path, 'r') as f:
        for line in f:
            img_name, classes = line[:-1].split(',')
            img_dict.append(img_name + ';' + str(classes))
    return img_dict

class ImageDataset(data.Dataset):
    def __init__(self, args, phase=None):
        args.num_classes = 100
        self.args =args
        self.root = '/home2/Datasets/' + args.root
        self.classes_list = self.root + '/' + 'classes.txt'
        self.phase = args.phase if phase == None else phase
        if self.phase == 'train':
            self.image_list_path = self.root + '/' + 'train_images.txt'
        elif self.phase == 'test':
            self.image_list_path = self.root + '/' + 'test_images.txt'
        self.crop_size = args.crop_size 
        self.resize_size = args.resize_size
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
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        self.label_classes = {} 
        with open(self.classes_list, 'r') as f:
            for i, line in enumerate(f):
                self.label_classes[line[:-1]] = i
        self.img_list = load_img_list(self.image_list_path)

    def __getitem__(self, index):
        if self.phase == 'train':
            img_name, img_class = self.img_list[index].split(';')
            path = self.root + '/'  + img_name  
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            img_class = int(img_class)
            return path, img, img_class
        else:
            img_name, img_class = self.img_list[index].split(';')
            path = self.root + '/'  + img_name 
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
            img_class = int(img_class)
            return img, img, img_class, img, path 

    def __len__(self):
        return len(self.img_list)