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

from src.utils import num_flat_features


##################
#  GENERATPORS   #
##################

def get_cnn(filters=[512]):
    prev = 1

    model = []

    for fil in filters:
        model.append(nn.Conv2d(prev, fil, 3))
        model.append(nn.ReLU())
        model.append(nn.MaxPool2d(2))
        prev = fil

    return nn.Sequential(*model)


#############
#  LAYERS   #
#############

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ConvMaxPool(nn.Module):
    def __init__(self, conv_in, conv_out, conv_ker, pool_ker):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv2d(conv_in, conv_out, conv_ker)
        self.pool = nn.MaxPool(pool_ker)

    def forward(self, x):
        return self.pool(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate=12):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class TransitionLayer(nn.Module):
    def __init__(self, nChannels, reduction):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        nOutChannels = int(math.floor(nChannels*reduction))
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nChannels, growthRate, depth, bottleneck):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(int(depth)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


###############################################################################
###                             NEW CLASS                                   ###
###############################################################################


class OrdCNN(nn.Module):
    def __init__(self):
        super(OrdCNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 64, 3)
        self.conv4 = nn.Conv2d(64, 32, 3)
        self.conv5 = nn.Conv2d(32, 16, 3)
        self.conv6 = nn.Conv2d(16, 8, 3)
        self.conv7 = nn.Conv2d(8, 4, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(64, 32) 
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        print(x.size())
        out = F.relu(self.conv7(x))
        print(out.size())
        x = out.view(-1, num_flat_features(out))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



###############################################################################
###                             NEW CLASS                                   ###
###############################################################################



class DenseNet(nn.Module):
    def __init__(self, growthRate, nLayers, nFc, reduction=0.5, nClasses=2, 
                        crosscon=False, bottleneck=True, max_pool=False):
        super(DenseNet, self).__init__()
        self.max_pool = max_pool

        # First convolution layer
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1, stride=2, bias=False)

        # DenseBlocks + TransitionLayers 
        main_blocks = []
        for i, depth in enumerate(nLayers):
            main_blocks.append(DenseBlock(nChannels, growthRate, depth, bottleneck))
            nChannels += depth * growthRate
            # Final Dense layer without transition
            if i != len(nLayers) - 1:
                main_blocks.append(TransitionLayer(nChannels, reduction))
                nChannels = int(math.floor(nChannels*reduction))
        self.main_blocks = nn.Sequential(*main_blocks)

        # Classification part (FC after global pooling)
        self.bn = nn.BatchNorm2d(nChannels)

        fc_part = []
        for i in range(len(nFc) - 1):
            fc_part.append(nn.Linear(nFc[i], nFc[i+1]))
            fc_part.append(nn.ReLU())
        self.fc_part = nn.Sequential(*fc_part)

        self.clf = nn.Linear(nFc[-1], nClasses)


    def forward(self, x):
        out = F.max_pool2d(self.conv1(x), 3, stride=2)
        out = self.main_blocks(out)
        
        out = self.bn(out)
        # it is a global pooling, so the resolution of the image after all previous blocks should be 7x7
        # is it ok for our part to have an avg pooling in the end? 
        #or 7 if it is previous models
        features = F.avg_pool2d(out, 3)


        #for skip connections:
        # mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        # mem_data = torch.cat((x, mem_data), 1)

        out = features.view(features.size(0), -1)
        # There was not anything about ReLu and BN in the original paper
        out = self.fc_part(out)
        out = self.clf(out)
        return out, features




###############################################################################
###                             NEW CLASS                                   ###
###############################################################################


class CustomMIL(nn.Module):
    def __init__(self, feature_extr=None, p2in=50*4*4, L=500, D=128, K=1, dropp=0.5):
        super(CustomMIL, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1
        self.p2in = p2in

        if feature_extr is not None:
            self.feature_extractor_part1 = feature_extr
        else:
            self.feature_extractor_part1 = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.p2in, self.L),
            nn.ReLU(),
            #nn.Dropout(dropp),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            #nn.Dropout(dropp)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.p2in)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        #return Y_prob, Y_hat, A
        return Y_prob




###############################################################################
###                             NEW CLASS                                   ###
###############################################################################



class DenseMIL(nn.Module):
    def __init__(self, growthRate, nLayers, reduction=0.5, nClasses=2, 
                        crosscon=False, bottleneck=True, p2in=50*4*4):
        super(DenseMIL, self).__init__()

        self.L = 500
        self.D = 128
        self.K = 1
        self.p2in = p2in

        # First convolution layer
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1, stride=2, bias=False)

        # DenseBlocks + TransitionLayers 
        main_blocks = []
        for i, depth in enumerate(nLayers):
            main_blocks.append(DenseBlock(nChannels, growthRate, depth, bottleneck))
            nChannels += depth * growthRate
            # Final Dense layer without transition
            if i != len(nLayers) - 1:
                main_blocks.append(TransitionLayer(nChannels, reduction))
                nChannels = int(math.floor(nChannels*reduction))
        self.main_blocks = nn.Sequential(*main_blocks)

        self.bn = nn.BatchNorm2d(nChannels)

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(self.p2in, self.L),
            nn.ReLU(),
            #nn.Dropout(dropp),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            #nn.Dropout(dropp)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            #nn.Sigmoid()
        )
    
    def freeze_weights(self):
        ## Freeze the gradients (it worse to check it again)
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.main_blocks.parameters():
            param.requires_grad = False
        for param in self.bn.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = x.squeeze(0)
        out = F.max_pool2d(self.conv1(x), 3, stride=2)
        out = self.main_blocks(out)
        
        out = self.bn(out)
        # it is a global pooling, so the resolution of the image after all previous blocks should be 7x7
        # is it ok for our part to have an avg pooling in the end? 
        #or 7 if it is previous models
        H = F.avg_pool2d(out, 3)
        
        H = H.view(-1, self.p2in)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        #return Y_prob, Y_hat, A
        return Y_prob