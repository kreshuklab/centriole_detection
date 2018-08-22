#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse

#INTERRNAL IMPORTS
from src.architectures import DenseNet, DenseNetSC
from src.datasets import CentriollesDatasetPatients
from src.utils import get_basic_transforms, log_info

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
    parser = argparse.ArgumentParser(description='Run learning of densnet implementation')

    parser.add_argument('--SC', action='store_true', help='Scip connections in NN')
    parser.add_argument('--BN', action='store_true', help='Bottle neck')
    parser.add_argument('--GR', type=int, default=12, help='Growth rate')
    parser.add_argument('--red', type=float, default=0.8, help='Reduction in number of channels')
    parser.add_argument('--depth', type=int, default=50, help='Depth of the denseblocks')

    parser.add_argument('--img_size', type=int, default=512, help='Save model weights each n epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, default=1e-6, help='Weight decey')
    parser.add_argument('--epoch', type=int, default=200, help='Number of epoches')

    parser.add_argument('--save_each', type=int, default=0, help='Save model weights each n epochs')
    parser.add_argument('--save_best', action='store_true', help='Save best test model?')
    parser.add_argument('--net_id', type=str, default='', help='Unique net id to save')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))

    # DATASETS INITIALIZATION
    train_tr, test_tr = get_basic_transforms()
    train_ds = CentriollesDatasetPatients(transform=train_tr, inp_size=args.img_size)
    test_ds  = CentriollesDatasetPatients(transform=test_tr,  inp_size=args.img_size, train=False)

    train_dl = DataLoader(train_ds,  batch_size=1, shuffle=True, num_workers=3)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=3)

    log_info('Datasets are initialized')

    # MODEL INITIALIZATION
    if args.SC:
        model = DenseNetSC(growthRate=args.GR, depth=args.depth, reduction=args.red, bottleneck=args.BN, nClasses=2)
    else:
        model = DenseNet (growthRate=args.GR, depth=args.depth, reduction=args.red, bottleneck=args.BN, nClasses=2)
    
    # TRAINING SETTINGS
    trainer = Trainer(model)
    trainer.bind_loader('train', train_dl).bind_loader('test', test_dl)

    if args.net_id != '':
        args.net_id = '_' + args.net_id
    weight_dir = 'models/densenet{}/weights'.format(args.net_id)
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

    logs_dir = 'models/densenet{}/logs'.format(args.net_id)
    trainer.build_logger(TensorboardLogger(log_scalars_every=(1, 'epochs'),
                                           log_images_every=(10, 'epochs')),
                                           log_directory=logs_dir)
    log_info('Logs will be written to {}'.format(logs_dir))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_info('Will be used : {}'.format(device))
    if device == 'cpu':
        trainer.cuda()
    
    ## TRAINING
    trainer.fit()



