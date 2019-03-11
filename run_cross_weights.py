#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

# BASIC IMPORTS
import argparse
import os
import subprocess
import sys

# INTERNAL IMPORTS
from src.datasets import CentriollesDatasetPatients
from src.utils import log_info, get_basic_transforms
from src.utils import init_weights
import src.implemented_models as impl_models

# INFERNO IMPORTS
import torch
from inferno.trainers.basic import Trainer
from torch.utils.data import DataLoader
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger
from inferno.trainers.callbacks.scheduling import AutoLR


def log_hist(self, tag, values=1, step=1, bins=1000):
    """Logs the histogram of a list/vector of values."""
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--fold', type=int, default=0, help='Number of fold for train/test split')

    parser.add_argument('--continue_training', action='store_true',
                        help='It is also nessesary to specify init_weights_path')
    parser.add_argument('--init_model_path', type=str,
                        default='../centrioles/run_history/ICL_DenseNet_3fc/true_save/weights/',
                        help='Name of the model for initialization')

    parser.add_argument('--freeze', action='store_true', help='Freezes first part')
    parser.add_argument('--check', action='store_true',
                        help='Specify this flag if you want to check that this code works')

    args = parser.parse_args()
    log_info('Params: ' + str(args))

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

    # Dataset
    train_tr, test_tr = get_basic_transforms()
    train_ds = CentriollesDatasetPatients(nums=train_folds[args.fold],
                                          main_dir='../centrioles/dataset/new_edition/combined',
                                          all_data=True,
                                          transform=train_tr, inp_size=512, train=True, check=args.check)
    test_ds = CentriollesDatasetPatients(nums=test_folds[args.fold],
                                         main_dir='../centrioles/dataset/new_edition/combined',
                                         all_data=True,
                                         transform=test_tr, inp_size=512, train=False, check=args.check)

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds,  batch_size=4, shuffle=True, num_workers=0)
    log_info('Datasets are initialized!')

    # MODEL MIL_DenseNet_3fc
    exec("model = impl_models.%s" % (args.model_name))

    if args.continue_training:
        trainer = Trainer(model)
        if torch.cuda.is_available():
            trainer = trainer.load(from_directory=args.init_model_path, best=True)
        else:
            trainer = trainer.load(from_directory=args.init_model_path, best=True, map_location='cpu')
        init_model = trainer.model
        init_model.features_needed = True

        init_weights(model, init_model, freeze_gradients=args.freeze)
    else:
        icl_model = impl_models.ICL_DenseNet_3fc
        path_to_model_weights = '../centrioles/run_history/ICL_DenseNet_3fc/true_save/weights/'
        trainer = Trainer(icl_model)
        if torch.cuda.is_available():
            trainer = trainer.load(from_directory=path_to_model_weights, best=True)
        else:
            trainer = trainer.load(from_directory=path_to_model_weights, best=True, map_location='cpu')
        icl_model = trainer.model
        icl_model.features_needed = True

        init_weights(model, icl_model, freeze_gradients=args.freeze,
                     filter=lambda x: 'main_blocks' in x or 'conv1' in x)

    # DIRS
    model_dir = os.path.join('models', args.model_name)
    curent_model_dir = os.path.join(model_dir, args.id)
    log_info('Model will be saved to %s' % (curent_model_dir))
    log_info(' + Number of params: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    weight_dir = os.path.join(curent_model_dir, 'weights')
    log_info('Weights will be saved to %s' % (weight_dir))
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    logs_dir = os.path.join(curent_model_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    log_info('Logs will be saved to %s' % (logs_dir))

    # Build trainer
    logger = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                               log_images_every=None,
                               log_histograms_every=None)

    def log_hist(self, tag, values=1, step=1, bins=1000):
        """Logs the histogram of a list/vector of values."""
        pass
    logger.log_histogram = log_hist

    trainer = Trainer(model)\
        .build_criterion('CrossEntropyLoss') \
        .build_metric('CategoricalError') \
        .build_optimizer('Adam') \
        .validate_every((1, 'epochs')) \
        .save_every((1, 'epochs')) \
        .save_to_directory(weight_dir) \
        .set_max_num_epochs(10000) \
        .build_logger(logger, log_directory=logs_dir) \
        .register_callback(AutoLR(0.96, (1, 'epochs'), monitor_momentum=0.9,
                           monitor_while='validating',
                           consider_improvement_with_respect_to='bests'))

    # Bind loaders
    trainer \
        .bind_loader('train', train_dl) \
        .bind_loader('validate', test_dl)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()
