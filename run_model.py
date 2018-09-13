#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse
import os
import subprocess
import sys

#INTERNAL IMPORTS
from src.datasets import CentriollesDatasetPatients, CentriollesDatasetBags, MnistBags
from src.utils import get_basic_transforms, log_info
from src.trainer import  train, validate
import src.implemented_models as impl_models

#TORCH IMPORTS
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

#INFERNO IMPORTS
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

from tensorboardX import SummaryWriter

###################
#    MAIN CODE    #
###################

if __name__ == "__main__":

    # ARGPUMENTS
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--use_bags', action='store_true', help='For MIL models images should be represented as bag')
    parser.add_argument('--img_size', type=int, default=512, help='Size of input images')
    parser.add_argument('--wsize', type=int, default=28, help='Size of windows for bagging')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--ld', type=float, default=0.95, help='Learning rate multipliyer for every 10 epoches')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epoch', type=int, default=0, help='Number of epoches')
    parser.add_argument('--test', action='store_true', help='Test this model on simpler dataset')
    parser.add_argument('--crop', action='store_true', help='Crop only the central cell')
    parser.add_argument('--stride', type=float, default=0.5, help='From 0 to 1')
    parser.add_argument('--pyramid_layers', type=int, default=28, help='Number of layers in da pyramid')

    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--save_each', type=int, default=0, help='Save model weights each n epochs')
    parser.add_argument('--save_best', action='store_true', help='Save best test model?')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))
    #log_info('GIT revision: ' + subprocess.check_output('git rev-parse HEAD', shell=True).decode("utf-8"))

    # DATASETS INITIALIZATION
    train_tr, test_tr = get_basic_transforms()
    if args.use_bags:
        if args.test: 
            train_ds = MnistBags(wsize=(args.wsize, args.wsize))
            test_ds  = MnistBags(wsize=(args.wsize, args.wsize), train=False)
            log_info('Test bags dataset is used')
        else:
            train_ds = CentriollesDatasetBags(transform=train_tr, 
                                                inp_size=args.img_size, wsize=(args.wsize, args.wsize), 
                                                crop=args.crop, stride=args.stride, pyramid_layers=args.pyramid_layers)
            test_ds  = CentriollesDatasetBags(transform=test_tr, 
                                                inp_size=args.img_size, wsize=(args.wsize, args.wsize), 
                                                crop=args.crop, stride=args.stride, train=False, 
                                                pyramid_layers=args.pyramid_layers)
            log_info('Bags dataset is used')
            #TODO: Average bag size

    else:
        if args.test:
            train_ds = CentriollesDatasetOn(transform=train_tr, pos_dir='dataset/0_cifar_class', neg_dir='dataset/0_cifar_class', inp_size=args.img_size)
            test_ds  = CentriollesDatasetOn(transform=test_tr , pos_dir='dataset/1_cifar_class', neg_dir='dataset/1_cifar_class', inp_size=args.img_size)
            log_info('Test bags dataset is used')
        else:
            train_ds = CentriollesDatasetPatients(transform=train_tr, inp_size=args.img_size)
            test_ds  = CentriollesDatasetPatients(transform=test_tr, inp_size=args.img_size, train=False)
            log_info('Patients dataset is used')  


    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=0)

    log_info('Datasets are initialized!')
    if not (args.use_bags and args.test):
        log_info('Train: size %d balance %f' % (len(train_ds), train_ds.class_balance()))
        log_info('Test : size %d balance %f' % (len(test_ds ), test_ds.class_balance() ))

    if args.use_bags and not args.test:
        bags_size = 0
        for i  in range(len(train_ds)):
            bags_size += train_ds[i][0].size()[0]
        bags_size /= len(train_ds)
        log_info('Mean bag size %f' % (bags_size))

    # MODEL INITIALIZATION
    model_dir = os.path.join('models', args.model_name)
    curent_model_dir = os.path.join(model_dir, args.id)
    exec("model = impl_models.%s" % (args.model_name))
    log_info('Model will be saved to %s' % (curent_model_dir))
    log_info('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    # TRAINING SETTINGS
    weight_dir = os.path.join(curent_model_dir, 'weights')
    log_info('Weights will be saved to %s' % (weight_dir))
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    logs_dir = os.path.join(curent_model_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    log_info('Logs will be saved to %s' % (logs_dir))

    if torch.cuda.is_available():
        log_info('Cuda will be used')
        device = torch.device("cuda:0")
    else:
        log_info('Cuda was not found, using CPU')
        device = torch.device("cpu")

    model.to(device)
    sys.stdout.flush()

    writer = SummaryWriter(log_dir=logs_dir)

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-3)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd ,betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.ld)
    
    ## Save examples of input
    # for img in train_dl:
    #     writer.add_image('Train', img[0], 0)
    #     break
    # for img in test_dl:
    #     writer.add_image('Test', img[0], 0)
    #     break

    ## TRAINING
    epoch_num = 0
    best_loss  = 1e5
    while True:
        log_info('Epoch %d satrted' %(epoch_num))

        ###########
        ## TRAIN ##
        ###########

        model.train()
        criterion.train()

        global_loss = 0.0
    
        for inputs, label in train_dl:
            inputs, label = inputs.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            
            global_loss += loss.item()
        
        global_loss /= len(train_dl)
        loss = global_loss
        writer.add_scalar('train_loss', loss, epoch_num)
        writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch_num)
        
        ################
        ## VAlIDATION ##
        ################

        model.eval()
        criterion.eval()

        global_loss = 0.0
        accuracy    = 0.0 
        
        for inputs, label in test_dl:
            inputs, label = inputs.to(device), label.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, label)
            global_loss += loss.item()
            accuracy    += (round(float(F.softmax(outputs, dim=1)[0][0])) == float(label))
        
        global_loss /= len(test_dl)
        accuracy    /= len(test_dl)

        loss, acc = global_loss, accuracy
        writer.add_scalar('test_loss', loss, epoch_num)
        writer.add_scalar('test_accuracy', acc, epoch_num)

        ################
        ## SAVE&CHECK ##
        ################

        if args.epoch != 0 and epoch_num >= args.epoch:
            log_info('Max number of epochs is exceeded. Training is finished!')
            break
        
        ## Save model
        if args.save_each != 0 and epoch_num % args.save_each == 0:
            file_name = '{}.pt'.format(str(epoch_num))
            torch.save(model, os.path.join(weight_dir, file_name))
            log_info('Model was saved in epoch number %d with validation accuracy %f and loss %f' % 
                        (epoch_num, acc, loss))
        if loss < best_loss:
            best_loss = loss
            torch.save(model, os.path.join(weight_dir, 'best_weight.pt'))
            log_info('Model was saved as best on epoch number %d with validation accuracy %f and loss %f' % 
                        (epoch_num, acc, loss))

        epoch_num += 1
        scheduler.step()





