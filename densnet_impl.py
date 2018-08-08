import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math


class OrdCNN(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 200, 3)
        self.conv2 = nn.Conv2d(200, 150, 3)
        self.conv3 = nn.Conv2d(150, 100, 3)
        self.conv4 = nn.Conv2d(100, 80, 3)
        self.conv5 = nn.Conv2d(80, 60, 3)
        self.conv6 = nn.Conv2d(60, 40, 3)
        self.conv7 = nn.Conv2d(40, 20, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(20 * 2 * 2, 20)
        self.fc2 = nn.Linear(20, 8)
        self.fc3 = nn.Linear(8, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.max_pool2d(F.relu(self.conv5(x)), 2)
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        x = F.max_pool2d(F.relu(self.conv7(x)), 2)
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







###############################################################################
###                             NEW CLASS                                   ###
###############################################################################





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
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out

class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans4 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans5 = Transition(nChannels, nOutChannels)

        # nChannels = nOutChannels
        # self.dense6 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # nChannels += nDenseBlocks*growthRate
        # nOutChannels = int(math.floor(nChannels*reduction))
        # self.trans6 = Transition(nChannels, nOutChannels)

        # nChannels = nOutChannels
        # self.dense7 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # nChannels += nDenseBlocks*growthRate
        # nOutChannels = int(math.floor(nChannels*reduction))
        # self.trans7 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.denseF = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.fc1 = nn.Linear(380, 138)
        self.fc2 = nn.Linear(138, 24)
        self.fc3 = nn.Linear(24, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.trans4(self.dense4(out))
        out = self.trans5(self.dense5(out))
        # out = self.trans6(self.dense6(out))
        # out = self.trans7(self.dense7(out))
        out = self.denseF(out)
        out = F.avg_pool2d(F.relu(self.bn1(out)), 8)
        # print(out.size())
        out = out.view(-1, self.num_flat_features(out))
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features







###############################################################################
###                             NEW CLASS                                   ###
###############################################################################






class DenseNetSC(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
        super(DenseNetSC, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        memChannels = 1
        self.conv1 = nn.Conv2d(1, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        tmp = nChannels
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels+ memChannels, nOutChannels)
        memChannels += tmp

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        tmp = nChannels
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels + memChannels, nOutChannels)
        memChannels += tmp

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        tmp = nChannels
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans3 = Transition(nChannels + memChannels, nOutChannels)
        memChannels += tmp

        nChannels = nOutChannels
        self.dense4 = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        tmp = nChannels
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans4 = Transition(nChannels + memChannels, nOutChannels)
        memChannels += tmp

        nChannels = nOutChannels
        self.dense5 = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        tmp = nChannels
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans5 = Transition(nChannels + memChannels, nOutChannels)
        memChannels += tmp

        # nChannels = nOutChannels
        # self.dense6 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # nChannels += nDenseBlocks*growthRate
        # nOutChannels = int(math.floor(nChannels*reduction))
        # self.trans6 = Transition(nChannels, nOutChannels)

        # nChannels = nOutChannels
        # self.dense7 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck)
        # nChannels += nDenseBlocks*growthRate
        # nOutChannels = int(math.floor(nChannels*reduction))
        # self.trans7 = Transition(nChannels, nOutChannels)

        nChannels = nOutChannels
        self.denseF = self._make_dense(nChannels + memChannels, growthRate, nDenseBlocks, bottleneck)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels + memChannels)
        self.fc1 = nn.Linear(1156, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        mem_data = x
        x = self.conv1(mem_data)
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.trans1(self.dense1(mem_data))
        
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.trans2(self.dense2(mem_data))
        
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.trans3(self.dense3(mem_data))
        
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.trans4(self.dense4(mem_data))
        
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.trans5(self.dense5(mem_data))
       
        mem_data = F.upsample(mem_data, size=(x.size(2), x.size(3)), mode='bilinear')
        mem_data = torch.cat((x, mem_data), 1)
        x = self.denseF(mem_data)

        x = F.avg_pool2d(F.relu(self.bn1(x)), 8)
        # print(out.size())
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











