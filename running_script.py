#!/g/kreshuk/lukoianov/miniconda3/envs/centenv/bin/python3

import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from densnet_impl import DenseNet, DenseNetSC, OrdCNN

class CentriollesDatasetOn(Dataset):
    """Centriolles dataset."""

    def __init__(self, pos_dir='dataset/cropped_pos/',
                       neg_dir='dataset/cropped_neg/', 
                all_data=False, train=True, fold=0, out_of=1, transform=None, inp_size=512):
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
            im = Image.open(os.path.join(pos_dir, img_name))
            im.load()
            im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
            self.samples.append(im.copy())
            self.classes.append(1)
            im.close
            
        ## Negative samples
        for img_name in get_img_names(neg_dir):
            im = Image.open(os.path.join(neg_dir, img_name))
            im.load()
            im.thumbnail((inp_size, inp_size), Image.ANTIALIAS)
            self.samples.append(im.copy())
            self.classes.append(0)
            im.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.samples[idx]), self.classes[idx]
        return self.samples[idx], self.classes[idx]
    
    def class_balance(self):
        return np.sum(self.classes) / len(self.classes)


def detect_mean_std():
    all_data = CentriollesDatasetOn(all_data=True, transform=transforms.ToTensor(), inp_size=2048) 
    for elem in DataLoader(all_data, batch_size=len(all_data)):
        inputs, labels = elem
        tmp = torchvision.utils.make_grid(inputs)
        gme = tmp.mean()
        gstd = tmp.std()
    return gme, gstd


if __name__ == "__main__":
    print('INFO: Stats detection started')
    gme, gstd = detect_mean_std()
    print('INFO: Stats detection ended')

    final_tr = transforms.Compose([transforms.RandomRotation(180),
                               transforms.RandomVerticalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize((gme, ), (gstd, ))])

    train_ds = CentriollesDatasetOn(transform=final_tr) 
    test_ds  = CentriollesDatasetOn(transform=final_tr, train=False)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=3)


    net = DenseNetSC(growthRate=15, depth=50, reduction=0.5, bottleneck=True, nClasses=2)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will be used : ', device)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)

    train_loss_ar = []
    test_loss_ar = []
    best_test_loss = 10e8

    print('INFO: Learning had been started')

    for epoch in range(100):  # loop over the dataset multiple time
        running_loss = 0.0
        test_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            #show_from_batch(torchvision.utils.make_grid(inputs))
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()

        for i, data in enumerate(test_dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

        if(test_loss / len(test_dl) < best_test_loss):
            best_test_loss = test_loss
            torch.save(net, 'best_weight.pt')
        
        print('[%d, %3d] train_loss: %.5f test_loss: %.5f' % 
              (epoch + 1, i + 1, running_loss / len(train_dl), test_loss / len(test_dl)))

        train_loss_ar.append(running_loss / len(train_dl))
        test_loss_ar.append(test_loss / len(test_dl))
    print('Finished Training')


    correct = 0
    total = 0
    with torch.no_grad():
        for data in train_dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Train     : %.2f %%' % (100 * correct / total))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test     : %.2f %%' % (100 * correct / total))
    print("Const ans: %.2f %%" % (100 * test_ds.class_balance()) )

    plt.plot(train_loss_ar, label="train")
    plt.plot(test_loss_ar, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(train_loss_ar)))
    plt.savefig('learning_plot.png')


