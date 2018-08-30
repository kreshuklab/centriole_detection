#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse
import os
import sys

#INTERNAL IMPORTS
from src.datasets import CentriollesDatasetPatients, CentriollesDatasetBags
from src.utils import get_basic_transforms, log_info
from src.trainer import  train, validate
import src.implemented_models as impl_models

#TORCH IMPORTS
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

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
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--ld', type=float, default=0.9, help='Learning rate multipliyer for every 10 epoches')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--epoch', type=int, default=0, help='Number of epoches')

    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--save_each', type=int, default=0, help='Save model weights each n epochs')
    parser.add_argument('--save_best', action='store_true', help='Save best test model?')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))

    # DATASETS INITIALIZATION
    train_tr, test_tr = get_basic_transforms()
    if args.use_bags:
        train_ds = CentriollesDatasetBags(transform=train_tr, nums=[402], inp_size=args.img_size)
        test_ds  = CentriollesDatasetBags(transform=test_tr, nums=[402], inp_size=args.img_size, train=False)
        log_info('Bags dataset is used')
        #TODO: Average bag size
    else:
        train_ds = CentriollesDatasetPatients(transform=train_tr, nums=[402], inp_size=args.img_size)
        test_ds  = CentriollesDatasetPatients(transform=test_tr,  nums=[402], inp_size=args.img_size, train=False)
        log_info('Patients dataset is used')  

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=True, num_workers=3)

    log_info('Datasets are initialized!')
    log_info('Train: size %d balance %f' % (len(train_ds), train_ds.class_balance()))
    log_info('Test : size %d balance %f' % (len(test_ds ), test_ds.class_balance() ))

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
        loss = train(model, train_dl, criterion, optimizer)
        writer.add_scalar('train_loss', loss, epoch_num)
        writer.add_scalar('learning_rate', float(scheduler.get_lr()[0]), epoch_num)
        
        loss, acc = validate(model, test_dl, criterion)
        writer.add_scalar('test_loss', loss, epoch_num)
        writer.add_scalar('test_accuracy', acc, epoch_num)

        if args.epoch != 0 and epoch_num >= arg.epoch:
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





