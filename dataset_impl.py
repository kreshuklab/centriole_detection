import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models

import sys
import math
import os
from PIL import Image
import numpy as np


class CentriollesDatasetOn(Dataset):
    """Centriolles dataset."""

    def __init__(self, pos_dir='dataset/positives',
                       neg_dir='dataset/negatives', 
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=2048):
        """
        Args:
            pos_sample_dir (string): Path to the directory with all positive samples
            neg_sample_dir (string): Path to the directory with all negative samples
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.samples = []
        self.classes = []
        self.transform = transform
        
        def get_img_names(dir_name):
            img_names = [f for f in os.listdir(dir_name) if f.endswith('.png')]
            if all_data:
                return img_names
            if out_of == 1:
                delimetr = int(0.6 * len(img_names))
            else:
                delimetr = int((fold + 1)/out_of * len(img_names))
            if train:
                img_names = img_names[:delimetr]
            else:
                img_names = img_names[delimetr:]
            return img_names

        
        ## Positive samples
        for img_name in get_img_names(pos_dir):
            im = Image.open(os.path.join(pos_dir, img_name)).convert('L')
            im.load()
            im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
            self.samples.append(im.copy())
            self.classes.append(1)
            im.close
            
        ## Negative samples
        for img_name in get_img_names(neg_dir):
            im = Image.open(os.path.join(neg_dir, img_name)).convert('L')
            im.load()
            im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
            self.samples.append(im.copy())
            self.classes.append(0)
            im.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx]).float(), self.classes[idx]
        return self.samples[idx].float(), self.classes[idx]
    
    def class_balance(self):
        return np.sum(self.classes) / len(self.classes)





###############################################################################
###                             NEW CLASS                                   ###
###############################################################################





class CentriollesDatasetPatients(Dataset):
    """Centriolles dataset."""

    def __init__(self, nums=[397, 3971, 402, 403, 406, 4021, 396], main_dir='dataset/new_edition/in_png',
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=512):
        self.samples = []
        self.classes = []
        self.patient = []
        self.transform = transform
        
        def get_img_names(dir_name):
            img_names = [f for f in os.listdir(dir_name) if f.endswith('.png')]
            if all_data:
                return img_names
            if out_of == 1:
                delimetr = int(0.6 * len(img_names))
            else:
                delimetr = int((fold + 1)/out_of * len(img_names))
            if train:
                img_names = img_names[:delimetr]
            else:
                img_names = img_names[delimetr:]
            return img_names

        
        ## Positive samples
        for num in nums:

            pos_dir = os.path.join(main_dir, str(num) + '_centrioles')
            neg_dir = os.path.join(main_dir, str(num) + '_nocentrioles')

            for img_name in get_img_names(pos_dir):
                im = Image.open(os.path.join(pos_dir, img_name)).convert('L')
                im.load()
                im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
                self.samples.append(im.copy())
                self.classes.append(1)
                self.patient.append(num)
                im.close

            ## Negative samples
            for img_name in get_img_names(neg_dir):
                im = Image.open(os.path.join(neg_dir, img_name)).convert('L')
                im.load()
                im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
                self.samples.append(im.copy())
                self.classes.append(0)
                self.patient.append(num)
                im.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx]).float(), self.classes[idx]
        return self.samples[idx].float(), self.classes[idx]
    
    def class_balance(self):
        return np.sum(self.classes) / len(self.classes)

    def class_balance_for_patients(self):
        positives = {}
        total     = {}
        for i, num in enumerate(self.patient):
            if num not in positives:
                positives[num] = 0.0
                total[num]     = 0.0
            positives[num] += self.classes[i]
            total[num]     += 1
        for num in positives:
            positives[num] = positives[num] / total[num]
        return positives

###############################################################################
###                             NEW CLASS                                   ###
###############################################################################


def image2bag(img, wsize=(28, 28), stride=0.5):
    bag=[]
    c, w, h = img.size()
    for cx in range(0, w - wsize[0], int(wsize[0] * stride)):
        for cy in range(0, h - wsize[1], int(wsize[1] * stride)):
            bag.append(img[:,cx:cx+wsize[0], cy:cy+wsize[1]])
    return bag


class CentriollesDatasetBags(Dataset):
    """Centriolles dataset."""

    def __init__(self, nums=[397, 3971, 402, 403, 406, 4021, 396], main_dir='dataset/new_edition/in_png',
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=512):
        self.samples = []
        self.classes = []
        self.patient = []
        self.transform = transform
        
        def get_img_names(dir_name):
            img_names = [f for f in os.listdir(dir_name) if f.endswith('.png')]
            if all_data:
                return img_names
            if out_of == 1:
                delimetr = int(0.6 * len(img_names))
            else:
                delimetr = int((fold + 1)/out_of * len(img_names))
            if train:
                img_names = img_names[:delimetr]
            else:
                img_names = img_names[delimetr:]
            return img_names

        
        ## Positive samples
        for num in nums:

            pos_dir = os.path.join(main_dir, str(num) + '_centrioles')
            neg_dir = os.path.join(main_dir, str(num) + '_nocentrioles')

            for img_name in get_img_names(pos_dir):
                im = Image.open(os.path.join(pos_dir, img_name)).convert('L')
                im.load()
                im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
                self.samples.append(im.copy())
                self.classes.append(1)
                self.patient.append(num)
                im.close

            ## Negative samples
            for img_name in get_img_names(neg_dir):
                im = Image.open(os.path.join(neg_dir, img_name)).convert('L')
                im.load()
                im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
                self.samples.append(im.copy())
                self.classes.append(0)
                self.patient.append(num)
                im.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            images, labels = self.transform(self.samples[idx]), self.classes[idx]
        else
            images, labels = self.samples[idx], self.classes[idx]
        images = image2bag(images.float())
        return images, labels

    
    def class_balance(self):
        return np.sum(self.classes) / len(self.classes)

    def class_balance_for_patients(self):
        positives = {}
        total     = {}
        for i, num in enumerate(self.patient):
            if num not in positives:
                positives[num] = 0.0
                total[num]     = 0.0
            positives[num] += self.classes[i]
            total[num]     += 1
        for num in positives:
            positives[num] = positives[num] / total[num]
        return positives





