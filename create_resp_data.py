#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

from src.implemented_models import ICL_DenseNet_3fc
from src.architectures import DenseNet
from inferno.trainers.basic import Trainer
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import numpy as np

import cv2
from src.datasets import CentriollesDatasetOn
from src.utils import get_basic_transforms, show_2d_slice
from torch.utils.data import DataLoader

from src.datasets import CentriollesDatasetBags, GENdataset
from src.utils import local_autoscale_ms

def get_resps_map(inp, out_size=28, features=False):
    img  = inp[0,:,:]
    mask = inp[1,:,:]
    mask = mask > mask.min()

    if features:
        reps = np.zeros((out_size, out_size, 1280))
    else:
        resps = np.zeros((out_size, out_size))
     
    wsize = (60, 60)
    stride = 512 / resps.shape[0] / wsize[0]
    th = 0.95
    color = (255, 0, 0)
    
    w, h = img.shape
    for i, cx in enumerate(range(0, w - wsize[0], int(wsize[0] * stride))):
        for j, cy in enumerate(range(0, h - wsize[1], int(wsize[1] * stride))):
            if mask[cx:cx+wsize[0], cy:cy+wsize[1]].sum() != wsize[0] * wsize[1]:
                continue
            cropped = local_autoscale_ms(img[cx:cx+wsize[0], cy:cy+wsize[1]])
            outputs, features = model(cropped[None, None, :, :].float())
            resps[i, j] =  int(float(F.softmax(outputs, dim=1)[0][1]) * 255)
    return resps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset of responce maps')

    parser.add_argument('--artif', action='store_true', help='Take artificial data')
    parser.add_argument('--features', action='store_true', help='Take artificial data')
    parser.add_argument('--out_size', type=int, default=28, help='Size of output image -> stride')

    args = parser.parse_args()

    trainer = Trainer(ICL_DenseNet_3fc)
    if torch.cuda.is_available():
        trainer = trainer.load(from_directory='../centrioles/models/ICL_DenseNet_3fc/true_save/weights/', best=True)
    else:
        trainer = trainer.load(from_directory='../centrioles/models/ICL_DenseNet_3fc/true_save/weights/', best=True, map_location='cpu')
    model = trainer.model

    train_tr, test_tr = get_basic_transforms()

    if args.artif:
        dataset = GENdataset(all_data=True, transform=test_tr, bags=False, crop=True)
        mid = 'artif'
    else:
        dataset = CentriollesDatasetBags(all_data=True, transform=test_tr, inp_size=512, bags=False, crop=True)
        mid = 'real' 

    if args.features:
        mid += '_features'

    dataloader = DataLoader(dataset, batch_size=1)

    for ind, (inp, label) in tqdm(enumerate(dataloader)):
        resp_map = get_resps_map(inp[0], features=args.features)
        if label[0]:
            cv2.imwrite('dataset/resp_map/' + mid + '/positive/' + str(ind) + '.png', resp_map)
        else:
            cv2.imwrite('dataset/resp_map/' + mid + '/negative/' + str(ind) + '.png', resp_map)
