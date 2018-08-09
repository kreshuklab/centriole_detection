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

from densnet_impl import DenseNet, DenseNetSC, OrdCNN, CentriollesDatasetOn


def detect_mean_std():
    all_data = CentriollesDatasetOn(all_data=True, transform=transforms.ToTensor(), inp_size=512) 
    for elem in DataLoader(all_data, batch_size=len(all_data)):
        inputs, labels = elem
        tmp = torchvision.utils.make_grid(inputs)
        gme = tmp.mean()
        gstd = tmp.std()
    return gme, gstd

def print_accuracy(net, dataloader, name):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, last_layer = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(name + ' : %.2f %%' % (100 * correct / total))


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


    #net = DenseNet(growthRate=12, depth=30, reduction=0.5, bottleneck=True, nClasses=2)
    net = OrdCNN()

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will be used : ', device)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-6, amsgrad=False)

    train_loss_ar = []
    test_loss_ar = []
    best_test_loss = 10e8


    print('INFO: Learning had been started')

    for epoch in range(400):  # loop over the dataset multiple time
        running_loss = 0.0
        test_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            #show_from_batch(torchvision.utils.make_grid(inputs))
            
            # forward + backward + optimize
            outputs, last_layer = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()

        for i, data in enumerate(test_dl, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, last_layer = net(inputs)
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


    print("Const ans: %.2f %%" % (100 * test_ds.class_balance()) )
    print()
    print_accuracy(net, train_dl, 'Last train')
    print_accuracy(net, test_dl,  'Last test ')
    net = torch.load('best_weight.pt')
    print()
    print_accuracy(net, train_dl, 'Final train')
    print_accuracy(net, test_dl,  'Final test ')


    plt.plot(train_loss_ar, label="train")
    plt.plot(test_loss_ar, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('learning_plot.png')


