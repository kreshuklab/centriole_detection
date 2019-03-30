'''
This module contains implementations of various
custom layers and architectures based on pytorch framework. 
'''

# BASIC IMPORTS
import sys
import math
import os
from PIL import Image
import numpy as np
from typing import List
from collections import OrderedDict

# TORCH IMPORTS
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models


def num_flat_features(x: torch.Tensor) -> int:
    '''
    Use this function to construct fully connected layer after squeezing.

    :param x:
        Torch tensor before squeezing
    :return:
        Number of fetures of the tensor after applying squeeze function.
    '''
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


##################
#  GENERATPORS   #
##################

def get_cnn(filters: List[int]=[512]) -> torch.nn.Sequential:
    '''
    Constructs Convolutional Neural Network with specified number of filters.
    Each block of returned model is:
        Conv2d(ker=3)
        ReLu()
        MaxPool2d(ker=2)

    :param filters:
        List of integers, representing number of output features for each block.
    :return:
        Pytorch model with architecture desribed above.
    '''

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
    '''
    Wrapper for method tensor.view()
    '''
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ConvMaxPool(nn.Module):
    '''
    Class of united Conv2d and MaxPool layers.
    Without activation!
    '''
    def __init__(self, conv_in, conv_out, conv_ker, pool_ker):
        super(ConvPool, self).__init__()
        self.conv = nn.Conv2d(conv_in, conv_out, conv_ker)
        self.pool = nn.MaxPool(pool_ker)

    def forward(self, x):
        return self.pool(self.conv(x))


class Bottleneck(nn.Module):
    '''
    Class of Bottleneck layer, describerd in the original papper of DenseNet.
    Basic unit for denseBlock, alternative to SingleLayer.
    Does not provide resolution reduction.
    It was shown, that this unit is more preferable.
    '''
    def __init__(self, nChannels: int, growthRate: int):
        '''
        :param nChannels:
            Number of input channels
        :param growthRate:
            Number of channels wich will be added to input.
            I.e. num of output channels = num of input + growthRate
        '''
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
    '''
    Class of SingleLayer, describerd in the original papper of DenseNet.
    Basic unit for denseBlock, alternative to Bottleneck layer.
    Does not provide resolution reduction.
    It was shown, that this unit is less preferable.
    '''
    def __init__(self, nChannels: int, growthRate: int=12):
        '''
        :param nChannels:
            Number of input channels
        :param growthRate:
            Number of channels wich will be added to input.
            I.e. num of output channels = num of input + growthRate
        '''
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class TransitionLayer(nn.Module):
    '''
    Class of TransitionLayer, describerd in the original papper of DenseNet.
    Provide connection between dense blocks and reduction of tensor resolution.
    '''
    def __init__(self, nChannels: int, reduction: float):
        '''
        :param nChannels:
            Number of input channels
        :param reduction:
            Ratio between output and input channels.
        '''
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
    '''
    Class of DenseBlock, describerd in the original papper of DenseNet.
    '''
    def __init__(self, nChannels: int, growthRate: int, depth: int, bottleneck: bool):
        '''
        :param nChannels:
            Number of input channels
        :param growthRate:
            Number of channels wich will be added to input in each basic unit.
            I.e. num of output channels = num of input + growthRate
        :param depth:
            Number of basic units in this DenseBlock.
        :param bottleneck:
            If True  -> basic unit is Bottleneck
            If False -> basic unit is SingleLayer
        '''
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
    '''
    Torch module wrapper for print.
    Prints size of a passed tensor.
    Used in debug.
    '''
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.size())
        return x


###############################################################################
#                                NEW CLASS                                    #
###############################################################################


class OrdCNN(nn.Module):
    '''
    Not flexible implementation of classical CNN.
    Better use construct_cnn function.
    '''
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
#                               NEW CLASS                                     #
###############################################################################


class DenseNet(nn.Module):
    '''
    Implementation of DenseNet.
    '''
    def __init__(self, growthRate: int, nLayers: List[int], nFc: List[int], reduction: float=0.5,
                 nClasses: int=2, crosscon: bool=False, bottleneck: bool=True, max_pool: bool=False,
                 inp_channels: int=1, features_needed: bool=False, drop_out_prob=None, avg_pool_size: int=3):

        '''
        :param growthRate:
            Number of channels wich will be added to input in each basic unit.
            I.e. num of output channels = num of input + growthRate
        :param nLayers:
            List specifying  depth of each DenseBlock.
            It will be constructed the same number of blocks as lenght of this list.
        :param nFc:
            List specifying input channels for each fully connected layer in
            the classification part of the DenseNet.
            It will be constructed the same number of fc layers as lenght of this list.
        :param reduction:
            Ratio between output and input channels in each Transition layer.
        :param inp_channels:
            Number of channels of input image.
        :param nClasses:
            Number of output classes.
        :param crosscon:
            Allows skip-connections between DenseBlocks (temprory disabled)
        :param bottleneck:
            If True  -> basic unit is Bottleneck
            If False -> basic unit is SingleLayer
        :param max_pool:
            Replace all average poolings with max pool (temprory disabled)
        :param features_needed:
            Drop features after last convolutional layer in the forward method?
        :param drop_out_prob:
            None (default if not needed), float - probability
        '''
        super(DenseNet, self).__init__()
        self.max_pool = max_pool
        self.avg_pool_size = avg_pool_size
        self.features_needed = features_needed

        # First convolution layer
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(inp_channels, nChannels, kernel_size=3, padding=1, stride=2, bias=False)

        # This change was made 5.03
        # self.features = nn.Sequential(OrderedDict([
        #             ('conv0', nn.Conv2d(inp_channels, nChannels, kernel_size=7, 
        #                                 stride=2, padding=3, bias=False)),
        #             ('norm0', nn.BatchNorm2d(nChannels)),
        #             ('relu0', nn.ReLU(inplace=True)),
        #             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        #         ]))

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
            if drop_out_prob is not None and drop_out_prob >= 0 and drop_out_prob <= 1:
                fc_part.append(nn.Dropout(p=drop_out_prob))

        self.fc_part = nn.Sequential(*fc_part)

        self.clf = nn.Linear(nFc[-1], nClasses)

    def forward(self, x):
        out = F.max_pool2d(self.conv1(x), 3, stride=2)
        out = self.main_blocks(out)

        out = self.bn(out)
        # it is a global pooling, so the resolution of the image after all previous blocks should be 7x7
        # is it ok for our part to have an avg pooling in the end?
        # or 7 if it is previous models
        features = F.avg_pool2d(out, self.avg_pool_size)

        # for skip connections:
        # mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        # mem_data = torch.cat((x, mem_data), 1)

        out = features.view(features.size(0), -1)
        # There was not anything about ReLu and BN in the original paper
        out = self.fc_part(out)
        out = self.clf(out)

        if self.features_needed:
            return out, features
        else:
            return out


###############################################################################
#                               NEW CLASS                                     #
###############################################################################


class FConvDenseNet(nn.Module):
    '''
    Implementation of FullyConvolutionalDenseNet.
    '''
    def __init__(self, growthRate: int, nLayers: List[int], nFc: List[int], reduction: float=0.5,
                 nClasses: int=2, crosscon: bool=False, bottleneck: bool=True, max_pool: bool=False,
                 inp_channels: int=1, features_needed: bool=False, drop_out_prob=None, avg_pool_size: int=3):
        super(FConvDenseNet, self).__init__()
        self.max_pool = max_pool
        self.avg_pool_size = avg_pool_size
        self.features_needed = features_needed

        # First convolution layer
        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(inp_channels, nChannels, kernel_size=3, padding=1, stride=2, bias=False)

        # This change was made 5.03
        # self.features = nn.Sequential(OrderedDict([
        #             ('conv0', nn.Conv2d(inp_channels, nChannels, kernel_size=7, 
        #                                 stride=2, padding=3, bias=False)),
        #             ('norm0', nn.BatchNorm2d(nChannels)),
        #             ('relu0', nn.ReLU(inplace=True)),
        #             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        #         ]))

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
            fc_part.append(nn.Conv2d(nFc[i], nFc[i+1], kernel_size=1))
            fc_part.append(nn.ReLU())
            if drop_out_prob is not None and drop_out_prob >= 0 and drop_out_prob <= 1:
                fc_part.append(nn.Dropout(p=drop_out_prob))

        self.fc_part = nn.Sequential(*fc_part)

        self.clf = nn.Conv2d(nFc[-1], nClasses, kernel_size=1)

    def forward(self, x):
        out = F.max_pool2d(self.conv1(x), 3, stride=2)
        out = self.main_blocks(out)

        out = self.bn(out)
        # it is a global pooling, so the resolution of the image after all previous blocks should be 7x7
        # is it ok for our part to have an avg pooling in the end?
        # or 7 if it is previous models
        features = F.avg_pool2d(out, self.avg_pool_size)
        out = features
        out = self.fc_part(out)
        out = self.clf(out)
        features = out
        out = F.max_pool2d(out, out.shape[2:]).view(out.shape[0])
        out = F.sigmoid(out)

        if self.features_needed:
            return out, features
        else:
            return out


###############################################################################
#                               NEW CLASS                                     #
###############################################################################


class CustomMIL(nn.Module):
    '''
    Implementation of MIL model with attention mechanism.
    '''
    def __init__(self, feature_extr: torch.nn.Module=None, p2in: int=50*4*4, dropp: int=0.5, L=500, D=128, K=1):
        '''
        :param feature_extr:
            Feature extraction part. For example convolution layers.
            If None -> default feature extracor, decribed in the paper will be used.
        :param p2in:
            Number of input channels after feature extraction part.
        :param dropp:
            Probability in dropuot layers (temprorary disabled).
        '''
        super(CustomMIL, self).__init__()
        self.L = L
        self.D = D
        self.K = K
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
            # nn.Dropout(dropp),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            # nn.Dropout(dropp)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        H = H.view(-1, self.p2in)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        # here all magic happens
        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        # return Y_prob, Y_hat, A
        # return A to see responces
        return Y_prob


###############################################################################
#                               NEW CLASS                                     #
###############################################################################


class DenseMIL(nn.Module):
    '''
    Implementation of MIL model with pretrained DenseNet as feature extreactor part.
    Because of GPU RAM limitations, gradients in the DenseNet should be freezed.
    We pretrained them on the artificial instance lavel dataset.
    Here you should repeat the same architecture of feature extractor as in pretrained model.

    Do not forget to call freeze_gradients method before training!
    '''
    def __init__(self, growthRate: int, nLayers: List[int], reduction: float=0.5, nClasses: int=2,
                 crosscon: bool=False, bottleneck: bool=True, p2in: int=50*4*4):
        '''
        :param growthRate:
            Number of channels wich will be added to input in each basic unit.
            I.e. num of output channels = num of input + growthRate
        :param nLayers:
            List specifying  depth of each DenseBlock.
            It will be constructed the same number of blocks as lenght of this list.
        :param reduction:
            Ratio between output and input channels in each Transition layer.
        :param nClasses:
            Number of output classes.
        :param crosscon:
            Allows skip-connections between DenseBlocks (temprory disabled)
        :param bottleneck:
            If True  -> basic unit is Bottleneck
            If False -> basic unit is SingleLayer
        :param p2in:
            Number of input channels after feature extraction part.
        '''
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
            # nn.Dropout(dropp),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            # nn.Dropout(dropp)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 2),
            # nn.Sigmoid()
        )

    def freeze_weights(self):
        '''
        Freeze gradients in the feature extraction part.

        Do not forget to load pretrained weights before!
        Use function init_weights from src.utils
        '''
        # Freeze the gradients (it worse to check it again)
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
        # is it ok for our part to have an avg pooling in the end ?
        # or 7 if it is previous models
        H = F.avg_pool2d(out, 3)

        H = H.view(-1, self.p2in)
        H = self.feature_extractor_part2(H)  # NxL

        A = self.attention(H)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        # return Y_prob, Y_hat, A
        return Y_prob
