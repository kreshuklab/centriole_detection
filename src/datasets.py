import torch
import torch.utils.data as data_utils

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from src.utils import image2bag, get_the_central_cell_mask

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
            im.close()
            
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

    def __init__(self, nums=[397, 402, 403, 406, 396, 3971, 4021], main_dir='dataset/new_edition/in_png_normilized',
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
                delimetr = int(0.75 * len(img_names))
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
                im.close()

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



class CentriollesDatasetBags(Dataset):
    """Centriolles dataset."""

    def __init__(self, nums=[397, 402, 403, 406, 396, 3971, 4021], main_dir='dataset/new_edition/in_png_normilized',
                 all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=512, wsize=(32, 32), 
                 stride=0.5, crop=False, pyramid_layers=1):
        self.samples = []
        self.classes = []
        self.patient = []
        self.transform = transform
        self.wsize = wsize 
        self.stride = stride
        self.crop = crop
        self.pyramid_layers = pyramid_layers

        def get_img(img_name):
            im = Image.open(img_name).convert('L')
            im.load()
            im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)

            cp_img = im.copy()
            im.close()

            if crop:
                mask = Image.fromarray(get_the_central_cell_mask(cp_img, wsize=wsize[0]))
                rot_mask = Image.new('L', (inp_size, inp_size), (1))
                return Image.merge("RGB", [cp_img, mask, rot_mask])
            else:
                return cp_img

        def get_img_names(dir_name):
            img_names = [f for f in os.listdir(dir_name) if f.endswith('.png')]
            if all_data:
                return img_names
            if out_of == 1:
                delimetr = int(0.75 * len(img_names))
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

            ## Positive samples
            for img_name in get_img_names(pos_dir):
                img = get_img(os.path.join(pos_dir, img_name))
                self.samples.append(img)
                self.classes.append(1)
                self.patient.append(num)

            ## Negative sampless
            for img_name in get_img_names(neg_dir):
                img = get_img(os.path.join(neg_dir, img_name))
                self.samples.append(img)
                self.classes.append(0)
                self.patient.append(num)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            images, labels = self.transform(self.samples[idx]), self.classes[idx]
        else:
            images, labels = self.samples[idx], self.classes[idx]

        images, _ = image2bag(images.float(), size=self.wsize, stride=self.stride, crop=self.crop, pyramid_layers=self.pyramid_layers)
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
            loader = data_utils.DataLoader(datasets.MNIST('dataset/MIL_test',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('dataset/MIL_test',
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

        return bag, label[0].long()

