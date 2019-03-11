#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

# BASIC IMPORTS
import argparse
import os
import subprocess
import sys

# INTERNAL IMPORTS
from src.datasets import GENdatasetILC
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
    parser.add_argument('--fold', type=int, default=0, help='Number of fold for train/test split')

    args = parser.parse_args()
    log_info('Params: ' + str(args))

    train_tr, test_tr = get_basic_transforms()

    train_folds = [[397, 402, 403, 406, 396, 3971, 4021, 3960, 406183],
                   [4010, 402, 403, 406, 4090, 4021, 40311, 40318, 40918, 406180, 406183],
                   [4010, 402, 403, 3971, 3960, 40311, 40318, 40918, 406180, 406183],
                   [3970, 397, 396, 3971, 3960, 40311, 40318, 40918, 406180, 406183],
                   [3970, 397, 396, 3971, 3960, 40311, 4010, 402, 403]]

    test_folds = [[3970, 4010, 4090, 40311, 40318, 40918, 406180],
                  [3970, 397, 396, 3971, 3960],
                  [3970, 397, 396, 406, 4090, 4021],
                  [4010, 402, 403, 406, 4090, 4021],
                  [40318, 40918, 406180, 406183, 4021]]

    train_ds = GENdatasetILC(nums=train_folds[args.fold],
                             main_dir='../centrioles/dataset/new_edition/combined',
                             transform=train_tr, inp_size=512, one=True, crop=True, stride=0.1)

    test_ds = GENdatasetILC(nums=test_folds[args.fold],
                            main_dir='../centrioles/dataset/new_edition/combined',
                            transform=test_tr, inp_size=512, one=True, crop=True, stride=0.1)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds,  batch_size=32, shuffle=True, num_workers=0)

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

    trainer = Trainer(model)\
        .build_criterion('CrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam') \
        .validate_every((2, 'epochs')) \
        .save_every((5, 'epochs')) \
        .save_to_directory(weight_dir) \
        .set_max_num_epochs(10000) \
        .build_logger(logger, log_directory=logs_dir) \
        .register_callback(AutoLR(0.96, (1, 'epochs'), monitor_momentum=0.9,
                           monitor_while='validating',
                           consider_improvement_with_respect_to='bes'))

    # Bind loaders
    trainer \
        .bind_loader('train', train_dl) \
        .bind_loader('validate', test_dl)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()
