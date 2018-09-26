#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse
import os
import subprocess
import sys

#INTERNAL IMPORTS
from src.datasets import CentriollesDatasetBags, GENdataset, MnistBags
from src.utils import get_basic_transforms, log_info, init_weights
from src.architectures import DenseNet
from src.implemented_models import ICL_DenseNet_3fc, ICL_MIL_DS3fc

#INFERNO IMPORTS
import torch
from inferno.trainers.basic import Trainer
from torch.utils.data import DataLoader
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.base import Callback
from inferno.trainers.callbacks.scheduling import AutoLR

class GradChecker(Callback):
    def end_of_epoch(self, **_):
        for name, value in self.trainer.model.named_parameters():
            if value.grad is not None:
                self.trainer.logger.writer.add_histogram(name, value.data.cpu().numpy(),
                                          self.trainer.iteration_count)
                self.trainer.logger.writer.add_histogram(name + '/grad',
                                          value.grad.data.cpu().numpy(),
                                          self.trainer.iteration_count)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--arti', action='store_true', help='Teach on the artificial data')
    parser.add_argument('--test', action='store_true', help='Teach on the mnist data')
    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--img_size', type=int, default=512, help='Size of input images')
    parser.add_argument('--stride', type=float, default=0.5, help='From 0 to 1')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--wsize', type=int, default=60, help='Size of windows for bagging')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))

    train_tr, test_tr = get_basic_transforms()

    if args.test:
        train_ds = MnistBags(wsize=(args.wsize, args.wsize))
        test_ds  = MnistBags(wsize=(args.wsize, args.wsize), train=False)
        log_info('Minst dataset is used')
    elif args.arti:
        train_ds = GENdataset(transform=train_tr,
                                inp_size=args.img_size, wsize=(args.wsize, args.wsize), 
                                crop=True, stride=args.stride)
        test_ds  = GENdataset(transform=test_tr,
                                inp_size=args.img_size, wsize=(args.wsize, args.wsize),
                                crop=True, stride=args.stride, train=False)
        log_info('GEN data is used')
    else:
        train_ds = CentriollesDatasetBags(transform=train_tr, 
                                            inp_size=args.img_size, wsize=(args.wsize, args.wsize), 
                                            crop=True, stride=args.stride)
        test_ds  = CentriollesDatasetBags(transform=test_tr, 
                                            inp_size=args.img_size, wsize=(args.wsize, args.wsize), 
                                            crop=True, stride=args.stride, train=False)
        log_info('Bags dataset is used')
    
    print('Test output: ', train_ds[0][0].shape, train_ds[0][1])
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=0)

    log_info('Datasets are initialized!')

    ref_trainer = Trainer(ICL_DenseNet_3fc)
    if torch.cuda.is_available():
        ref_trainer = ref_trainer.load(from_directory='../centrioles/models/ICL_DenseNet_3fc/true_save/weights/',
                                       best=True)
    else:
        ref_trainer = ref_trainer.load(from_directory='../centrioles/models/ICL_DenseNet_3fc/true_save/weights/',
                                       best=True, map_location='cpu')
    ref_model = ref_trainer.model

    model = ICL_MIL_DS3fc
    init_weights(model, ref_model)
    model.freeze_weights()

    ### DIRS AND MODEL
    model_dir = os.path.join('models', args.model_name)
    curent_model_dir = os.path.join(model_dir, args.id)
    log_info('Model will be saved to %s' % (curent_model_dir))
    log_info('  + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    weight_dir = os.path.join(curent_model_dir, 'weights')
    log_info('Weights will be saved to %s' % (weight_dir))
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    logs_dir = os.path.join(curent_model_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    log_info('Logs will be saved to %s' % (logs_dir))


    # Build trainer
    trainer = Trainer(model) \
        .build_criterion('CrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam', lr=args.lr, betas=(0.9, 0.999), eps=1e-08) \
        .validate_every((1, 'epochs')) \
        .save_every((5, 'epochs')) \
        .save_to_directory(model_dir) \
        .set_max_num_epochs(10000) \
        .register_callback(GradChecker()) \
        .register_callback(AutoLR(0.9, (1, 'epochs'), 
                                    consider_improvement_with_respect_to='previous'))\
        .build_logger(TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                        log_images_every=(1, 'epoch')),
                                        log_directory=logs_dir)



    # Bind loaders
    trainer \
        .bind_loader('train', train_dl) \
        .bind_loader('validate', test_dl)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()