#! /g/kreshuk/lukoianov/miniconda3/envs/centenv/bin/python

import numpy as np
import os
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CentriollesDatasetOn(Dataset):
    """Centriolles dataset."""

    def __init__(self, pos_dir='dataset/cropped_pos/',
                       neg_dir='dataset/cropped_neg/', 
                all_data=False, train=True, fold=0, out_of=1, transform=None):
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
            self.samples.append(im.copy())
            self.classes.append(1)
            im.close
            
        ## Negative samples
        for img_name in get_img_names(neg_dir):
            im = Image.open(os.path.join(neg_dir, img_name))
            im.load()
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



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 22, 5)
        self.conv4 = nn.Conv2d(22, 30, 5)
        self.conv5 = nn.Conv2d(30, 32, 5)
        self.conv6 = nn.Conv2d(32, 30, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(28 * 28 * 30, 220)
        self.fc2 = nn.Linear(220, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



def detect_mean_std():
    all_data = CentriollesDatasetOn(all_data=True, transform=transforms.ToTensor()) 
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

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=4, shuffle=True, num_workers=3)


    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


    train_loss_ar = []
    test_loss_ar = []
    best_test_loss = 10e8

    print('INFO: Learning had been started')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will be used : ', device)
    net.to(device)

    for epoch in range(100):  # loop over the dataset multiple times
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
        for data in test_dl:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Test     : %.2f %%' % (100 * correct / total))
    print("Const ans: %.2f %%" % (100 * test_ds.class_balance()) )

    plt.figure(figsize=(16, 6))
    plt.plot(train_loss_ar, label="train")
    plt.plot(test_loss_ar, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks(range(len(train_loss_ar)))
    plt.savefig('learning_plot.png')


