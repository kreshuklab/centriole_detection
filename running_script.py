#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3
import numpy as np
import os
from PIL import Image
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import scipy.misc
from skimage.transform import resize as sk_resize
import sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import inferno.io.transform as inftransforms


from nets_impl import DenseNet, DenseNetSC, OrdCNN, AttentionMIL
from dataset_impl import CentriollesDatasetOn, CentriollesDatasetPatients


def detect_mean_std():
    all_data = CentriollesDatasetPatients(all_data=True, transform=transforms.ToTensor(), inp_size=512) 
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


def save_tensor(tensor, i, name, size=(2048,2048)):
    npimg = tensor[0,i,:,:].detach().numpy()
    tr = npimg
    tr -= tr.min()
    tr /= tr.max()
    resized = sk_resize(tr, size)
    scipy.misc.imsave(name + '.png', resized)

def save_last_conv_activations(net, lla, dataset, name):
    label = -1
    i = 0
    dataiter = iter(train_dl)
    while label != 0:
        inputs, labels = dataiter.next()
        label = labels[0]
        i += 1
        
    if not os.path.exists(name):
        os.makedirs(name)
    save_tensor(inputs, 0, name + '/original' + str(lla))
    x, outputs = net(inputs)
    for i in range(outputs.size()[1]):
        save_tensor(outputs, i, name + '/' + str(i))


if __name__ == "__main__":
    # print('INFO: Stats detection started')
    # sys.stdout.flush()
    # #gme, gstd = 128, 10
    # gme, gstd = detect_mean_std()
    print('INFO: Dataset loading started')
    sys.stdout.flush()

    train_tr = transforms.Compose([ transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply(
                                        [transforms.RandomAffine(degrees  =180,
                                                            translate=(0.1, 0.1),
                                                            scale    =(0.9, 1.0),
                                                            shear    =10)]),
                                    inftransforms.image.PILImage2NumPyArray(),
                                    inftransforms.image.ElasticTransform(alpha=100, sigma=50),
                                    inftransforms.generic.NormalizeRange(normalize_by=255.0),
                                    inftransforms.generic.AsTorchBatch(dimensionality=2)])

    test_tr  = transforms.Compose([ transforms.RandomVerticalFlip(),
                                    transforms.RandomHorizontalFlip(),
                                    inftransforms.image.PILImage2NumPyArray(),
                                    inftransforms.generic.NormalizeRange(normalize_by=255.0),
                                    inftransforms.generic.AsTorchBatch(dimensionality=2)])

    train_ds = CentriollesDatasetPatients(transform=train_tr)
    test_ds  = CentriollesDatasetPatients(transform=test_tr, train=False)

    train_dl = DataLoader(train_ds,  batch_size=1, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=3)

    print('Datasets Train:', len(train_ds), ' Test: ', len(test_ds))
    sys.stdout.flush()

    print()
    print('Train dataset balance: ', train_ds.class_balance())
    print('Test  dataset balance: ', test_ds.class_balance())
    sys.stdout.flush()
    print()
    print('Train dataset balance for patients: ', train_ds.class_balance_for_patients())
    print('Test  dataset balance for patients: ', test_ds.class_balance_for_patients())
    sys.stdout.flush()
    net = AttentionMIL()
    # net = DenseNet(growthRate=5, depth=46, reduction=0.5, bottleneck=True, nClasses=2)
    # net = OrdCNN()

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    sys.stdout.flush()


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Will be used : ', device)
    sys.stdout.flush()
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(net.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6, amsgrad=False)

    train_loss_ar = []
    test_loss_ar = []
    best_test_loss = 10e8


    print('INFO: Learning had been started')
    sys.stdout.flush()

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
    save_last_conv_activations(net, 1, train_ds, 'debug/train_1')
    save_last_conv_activations(net, 1, test_ds, 'debug/test_1')
    net = torch.load('best_weight.pt')
    print()
    print_accuracy(net, train_dl, 'Final train')
    print_accuracy(net, test_dl,  'Final test ')
    save_last_conv_activations(net, 1, train_ds, 'debug/final_train_1')
    save_last_conv_activations(net, 1, test_ds, 'debug/final_test_1')



    plt.plot(train_loss_ar, label="train")
    plt.plot(test_loss_ar, label="test")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('learning_plot.png')


