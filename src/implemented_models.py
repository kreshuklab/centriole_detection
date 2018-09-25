from src.architectures import View, DenseNet, CustomMIL, get_cnn, DenseMIL
from inferno.extensions.layers.convolutional import ConvELU2D
from inferno.extensions.layers.reshape import Flatten

import torch.nn as nn

#### MIL ####
#original (used to work on 28x28)
MIL_32x32_to4x4 = CustomMIL()

MIL_48x48_to4x4_cst = CustomMIL(feature_extr=get_cnn(filters=[512, 512, 512]),  p2in=512*4*4, L=500, D=128, K=1)
MIL_48x48_to4x4_dec = CustomMIL(feature_extr=get_cnn(filters=[512, 256, 128]),  p2in=128*4*4, L=500, D=128, K=1)
MIL_48x48_to4x4_inc = CustomMIL(feature_extr=get_cnn(filters=[128, 256, 512]),  p2in=512*4*4, L=500, D=128, K=1)
MIL_48x48_to4x4_small = CustomMIL(feature_extr=get_cnn(filters=[128, 64, 32]),  p2in=32*4*4, L=500, D=128, K=1)
MIL_48x48_to4x4_small_drop0 = CustomMIL(feature_extr=get_cnn(filters=[128, 64, 32]),  p2in=32*4*4, L=500, D=128, K=1, dropp=0)
MIL_48x48_to4x4_small_drop2 = CustomMIL(feature_extr=get_cnn(filters=[128, 64, 32]),  p2in=32*4*4, L=500, D=128, K=1, dropp=0.2)

#### DenseNets ####
DenseNet_BN_32k_to7x7_ap_3fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 100])
DenseNet_BN_32k_to7x7_mp_3fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 100], max_pool=True)
DenseNet_BN_32k_to7x7_mp_5fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 512, 128, 32], max_pool=True)

ICL_DenseNet_3fc = DenseNet(32, [6, 12, 32], [1280, 80, 16])
ICL_MIL_DS3fc    = DenseMIL(32, [6, 12, 32], p2in=1280)
ICL_DenseNet_4fc = DenseNet(32, [12, 22, 42], [1808, 320, 80, 4])

MyDenceMILFeatures = DenseNet(32, [6, 12], [512, 80, 16], inp_channels=1280, features_needed=False)
MyDenceMIL = DenseNet(32, [6, 12], [512, 80, 16], inp_channels=1, features_needed=False)

#### VGGS ###
CNN_512_1conv_to15x15_6fc_32filter = \
         nn.Sequential(nn.Conv2d(1, 128, 32), nn.MaxPool2d(32), View(),
                       nn.Linear(128 * 15 * 15, 128 * 4), nn.ReLU(),
                       nn.Linear(128 * 4, 128), nn.ReLU(),
                       nn.Linear(128, 64), nn.ReLU(),
                       nn.Linear(64, 32), nn.ReLU(),
                       nn.Linear(32, 8), nn.ReLU(),
                       nn.Linear(8, 2))

CNN_512_3conv_to12x12_3fc_filter = \
         nn.Sequential(nn.Conv2d(1, 128, 32), nn.MaxPool2d(16),
                       nn.Conv2d(128, 1024, 3), nn.MaxPool2d(2),
                       nn.Conv2d(1024, 128, 3), View(),
                       nn.Linear(128 * 12 * 12, 128 * 12), nn.ReLU(),
                       nn.Linear(128 * 12, 128), nn.ReLU(),
                       nn.Linear(128, 2))

CNN_512_4conv_to4x4_3fc_32filter = \
         nn.Sequential(nn.Conv2d(1, 128, 32), nn.MaxPool2d(16),
                       nn.Conv2d(128, 128, 3), nn.MaxPool2d(2),
                       nn.Conv2d(128, 64, 3), nn.MaxPool2d(2), 
                       nn.Conv2d(64, 64, 3), View(),
                       nn.Linear(64 * 4 * 4, 64 * 4), nn.ReLU(),
                       nn.Linear(64 * 4, 64), nn.ReLU(),
                       nn.Linear(64, 2))

CNN_512_5conv_to10x10_3fc_16filter = \
         nn.Sequential(nn.Conv2d(1, 64, 5), nn.MaxPool2d(2),
                       nn.Conv2d(64, 128, 16), nn.MaxPool2d(4),
                       nn.Conv2d(128, 64, 8), nn.MaxPool2d(2), 
                       nn.Conv2d(64, 64, 3), nn.MaxPool2d(2),
                       nn.Conv2d(64, 64, 3), View(),
                       nn.Linear(64 * 10 * 10, 64 * 10), nn.ReLU(),
                       nn.Linear(64 * 10, 64), nn.ReLU(),
                       nn.Linear(64, 2))

CNN_512_7conv_to4x4_3fc = \
         nn.Sequential(nn.Conv2d(1, 1024, 3), nn.MaxPool2d(2),
                       nn.Conv2d(1024, 512, 3), nn.MaxPool2d(2),
                       nn.Conv2d(512, 256, 3), nn.MaxPool2d(2),
                       nn.Conv2d(256, 128, 3), nn.MaxPool2d(2), 
                       nn.Conv2d(128, 64, 3), nn.MaxPool2d(2),
                       nn.Conv2d(64, 32, 3), nn.MaxPool2d(2),
                       nn.Conv2d(32, 16, 3), View(),
                       nn.Linear(16 * 4 * 4, 16 * 4), nn.ReLU(),
                       nn.Linear(16 * 4, 16), nn.ReLU(),
                       nn.Linear(16, 2))

Inferno_example = nn.Sequential(
    ConvELU2D(in_channels=1, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    ConvELU2D(in_channels=256, out_channels=256, kernel_size=3),
    nn.MaxPool2d(kernel_size=2, stride=2),
    Flatten(),
    nn.Linear(in_features=(256 * 3 * 3), out_features=2),
    nn.Softmax(dim=1)
)

