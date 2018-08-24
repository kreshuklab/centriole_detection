#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse
import os

#INTERNAL IMPORTS
from src.datasets import CentriollesDatasetPatients, CentriollesDatasetBags
from src.utils import get_basic_transforms, log_info
import src.implemented_models as impl_models

#TORCH IMPORTS
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

#INFERNO IMPORTS
from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger


###################
#    MAIN CODE    #
###################

if __name__ == "__main__":

    # ARGPUMENTS
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--img_size', type=int, default=512, help='Size of input images')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decey')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epoches')

    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--save_each', type=int, default=0, help='Save model weights each n epochs')
    parser.add_argument('--save_best', action='store_true', help='Save best test model?')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))

    # DATASETS INITIALIZATION
    train_tr, test_tr = get_basic_transforms()
    train_ds = CentriollesDatasetPatients(transform=train_tr, inp_size=args.img_size)
    test_ds  = CentriollesDatasetPatients(transform=test_tr,  inp_size=args.img_size, train=False)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=3)

    log_info('Datasets are initialized')

    # MODEL INITIALIZATION
    model_dir = os.path.join('models', args.model_name)
    curent_model_dir = os.path.join(model_dir, args.id)
    exec("model = impl_models.%s" % (args.model_name))
    
    # TRAINING SETTINGS
    trainer = Trainer(model)
    trainer.bind_loader('train', train_dl).bind_loader('validate', test_dl)

    weight_dir = os.path.join(curent_model_dir, 'weights')
    if args.save_each != 0:
        trainer.save_to_directory(weight_dir).save_every((args.save_each, 'epochs'))
        log_info('Weights will be saved each {} epochs at {}'.format(args.save_each, weight_dir))
    if args.save_best:
        trainer.save_to_directory(weight_dir).save_at_best_validation_score()
        log_info('Best weights will be saved at {}'.format(weight_dir))
    
    trainer.validate_every((1, 'epochs'))
    trainer.build_metric('CategoricalError')

    trainer.build_criterion(nn.CrossEntropyLoss)
    trainer.build_optimizer('Adam', lr=args.lr, weight_decay=args.wd)

    trainer.set_max_num_epochs(args.epoch)

    logs_dir = os.path.join(curent_model_dir, 'logs')
    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'epochs'),
                                           log_images_every=(10, 'epochs')),
                                           log_directory=logs_dir)
    log_info('Logs will be written to {}'.format(logs_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        log_info('Cuda will be used')
        trainer.cuda()
    else:
        log_info('Cuda was not found, using CPU')

    ## TRAINING
    trainer.fit()



