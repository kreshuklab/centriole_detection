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

    def __init__(self, nums=[397, 402, 403, 406, 396], main_dir='dataset/new_edition/in_png_normilized',
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=2048):
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
            cropped = img[:,cx:cx+wsize[0], cy:cy+wsize[1]]
            bag.append(cropped)
    return torch.stack(bag)



class CentriollesDatasetBags(Dataset):
    """Centriolles dataset."""

    def __init__(self, nums=[397, 402, 403, 406, 396], main_dir='dataset/new_edition/in_png_normilized',
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=512, wsize=(28, 28), stride=0.5):
        self.samples = []
        self.classes = []
        self.patient = []
        self.transform = transform
        self.wsize = wsize 
        self.stride = stride

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
        else:
            images, labels = self.samples[idx], self.classes[idx]
        images = image2bag(images.float(), wsize=self.wsize, stride=self.stride)
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



import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                   mean_bag_length=10,
                                                   var_bag_length=2,
                                                   num_bag=100,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(target_number=9,
                                                  mean_bag_length=10,
                                                  var_bag_length=2,
                                                  num_bag=100,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.min(len_bag_list_train), np.max(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.min(len_bag_list_test), np.max(len_bag_list_test)))


