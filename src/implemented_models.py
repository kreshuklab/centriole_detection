from src.architectures import View, DenseNet, CustomMIL

import torch.nn as nn


#### FEs for MIL ####
CNN_MIL_f100x100_t2x2_dec =  \
                nn.Sequential(  nn.Conv2d(1, 512, 3), nn.MaxPool2d(3),
                                nn.Conv2d(512, 258, 3), nn.MaxPool2d(3),
                                nn.Conv2d(258, 128, 3), nn.MaxPool2d(3))
CNN_MIL_f100x100_t2x2_inc =  \
                nn.Sequential(  nn.Conv2d(1, 128, 3), nn.MaxPool2d(3),
                                nn.Conv2d(128, 258, 3), nn.MaxPool2d(3),
                                nn.Conv2d(258, 512, 3), nn.MaxPool2d(3))
CNN_MIL_f100x100_t10x10_dec =  \
                nn.Sequential(  nn.Conv2d(1, 128, 3), nn.MaxPool2d(3),
                                nn.Conv2d(128, 64, 3), nn.MaxPool2d(3))

#### MIL ####
#original (used to work on 28x28)
MIL_32x32_to4x4 = CustomMIL()

MIL_32x32_to4x4_k64 = CustomMIL(K=64)
MIL_32x32_to4x4_k4 = CustomMIL(K=4)
MIL_32x32_to4x4_k4 = CustomMIL(K=2)

## should work on 2048x2048
MIL_f100_t2_dec(feature_extr=CNN_MIL_f100x100_t2x2_dec,  p2in=128*2*2, L=500, D=128, K=1)
MIL_f100_t2_inc(feature_extr=CNN_MIL_f100x100_t2x2_inc,  p2in=128*2*2, L=500, D=128, K=1)
MIL_f100_t2_dec(feature_extr=CNN_MIL_f100x100_t10x10_dec,  p2in=64*10*10, L=500, D=128, K=1)

MIL_f100_t2_dec_k4(feature_extr=CNN_MIL_f100x100_t2x2_dec,  p2in=128*2*2, L=500, D=128, K=2)
MIL_f100_t2_inc_k4(feature_extr=CNN_MIL_f100x100_t2x2_inc,  p2in=128*2*2, L=500, D=128, K=2)
MIL_f100_t2_dec_k4(feature_extr=CNN_MIL_f100x100_t10x10_dec,  p2in=64*10*10, L=500, D=128, K=2)

MIL_f100_t2_dec_k4(feature_extr=CNN_MIL_f100x100_t2x2_dec,  p2in=128*2*2, L=500, D=128, K=4)
MIL_f100_t2_inc_k4(feature_extr=CNN_MIL_f100x100_t2x2_inc,  p2in=128*2*2, L=500, D=128, K=4)
MIL_f100_t2_dec_k4(feature_extr=CNN_MIL_f100x100_t10x10_dec,  p2in=64*10*10, L=500, D=128, K=4)


#### DenseNets ####
DenseNet_BN_32k_to7x7_ap_3fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 100])
DenseNet_BN_32k_to7x7_mp_3fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 100], max_pool=True)
DenseNet_BN_32k_to7x7_mp_5fc = DenseNet(32, [6, 12, 32, 64, 48], [2880, 1440, 512, 128, 32], max_pool=True)


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

