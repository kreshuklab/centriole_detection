#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

# BASIC IMPORTS
import argparse
import os
import subprocess
import sys
import numpy as np

# INTERNAL IMPORTS
from src.datasets import GENdatasetILC, CentriollesDatasetPatients
from src.utils import get_basic_transforms, log_info, get_resps_transforms
import src.implemented_models as impl_models

# INFERNO IMPORTS
import torch
from inferno.trainers.basic import Trainer
from torch.utils.data import DataLoader
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--img_size', type=int, default=60, help='Size of input images')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')

    parser.add_argument('--rdv', action='store_true', help='Real data validation')
    parser.add_argument('--rdt', action='store_true', help='Fixed adjustable learning rate')
    parser.add_argument('--init_model_path', type=str, default='',
                        help='Name of the model for initialization')


    parser.add_argument('--flr', action='store_true', help='Fixed adjustable learning rate')
    parser.add_argument('--decey', type=float, default=0.96, help='lr decey')


    args = parser.parse_args()
    log_info('Params: ' + str(args))

    train_tr, test_tr = get_basic_transforms()

    if args.rdt:
        train_ds = CentriollesDatasetPatients(nums=[397, 402, 403, 406, 396, 3971, 4021, 3960, 406183],
                                             main_dir='../centrioles/dataset/new_edition/combined',
                                             transform=train_tr, inp_size=512, train=True)
    else:
        train_ds = GENdatasetILC(main_dir='../centrioles/dataset/new_edition/in_png_normilized/',
                                 transform=train_tr, inp_size=512, one=True, crop=True, stride=0.1)

    if args.rdv or args.rdt:
        test_ds = CentriollesDatasetPatients(nums=[397, 402, 403, 406, 396, 3971, 4021, 3960, 406183],
                                             main_dir='../centrioles/dataset/new_edition/combined',
                                             transform=test_tr, inp_size=512, train=False)
    else:
        test_ds = GENdatasetILC(main_dir='../centrioles/dataset/new_edition/in_png_normilized/',
                                transform=test_tr, inp_size=512, one=True, crop=True, stride=0.1)

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds,  batch_size=args.batch, shuffle=True, num_workers=0)

    log_info('Datasets are initialized!')

    # DIRS AND MODEL
    exec("model = impl_models.%s" % (args.model_name))

    model_dir = os.path.join('models', args.model_name)
    curent_model_dir = os.path.join(model_dir, args.id)
    log_info('Model will be saved to %s' % (curent_model_dir))
    log_info(' + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    weight_dir = os.path.join(curent_model_dir, 'weights')
    log_info('Weights will be saved to %s' % (weight_dir))
    if not os.path.exists(weight_dir):
        os.mkdir(weight_dir)
    logs_dir = os.path.join(curent_model_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    log_info('Logs will be saved to %s' % (logs_dir))

    # Build trainer 
    logger = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                               log_images_every=(np.inf, 'epochs'))

    def log_histogram(self, tag, values, bins=1000):
        pass
    logger.log_histogram = log_histogram


    trainer = Trainer(model)

    if args.init_model_path != '':
        if torch.cuda.is_available():
            trainer = trainer.load(from_directory=args.init_model_path, best=True)
        else:
            trainer = trainer.load(from_directory=args.init_model_path, best=True, map_location='cpu')        

    trainer = trainer.\
            .build_criterion('BCELoss') \
            .build_metric('CategoricalError') \
            .build_optimizer('Adam') \
            .validate_every((2, 'epochs')) \
            .save_every((5, 'epochs')) \
            .save_to_directory(weight_dir) \
            .set_max_num_epochs(10000) \
            .build_logger(logger, log_directory=logs_dir)

    if args.flr:
        trainer = trainer.register_callback(AutoLR(args.decey, (1, 'epochs'), monitor_momentum=0.9,
                                            monitor_while='validating',
                                            consider_improvement_with_respect_to='best'))
    else:
        trainer = trainer.register_callback(AutoLR(0.9, (1, 'epochs'),
	                                        consider_improvement_with_respect_to='previous'))


    # Bind loaders
    trainer \
        .bind_loader('train', train_dl) \
        .bind_loader('validate', test_dl)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()
