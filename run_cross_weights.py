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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--check', action='store_true',
                        help='Specify this flag if you want to check that this code works')

    args = parser.parse_args()
    log_info('Params: ' + str(args))

    # Dataset
    train_tr, test_tr = get_basic_transforms()
    train_ds = CentriollesDatasetPatients(transform=train_tr, inp_size=512, train=True, check=args.check)
    test_ds = CentriollesDatasetPatients(transform=test_tr, inp_size=512, train=False, check=args.check)

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds,  batch_size=4, shuffle=True, num_workers=0)
    log_info('Datasets are initialized!')

    # MODEL
    model = impl_models.MIL_DenseNet_3fc
    model_name = 'MIL_DenseNet_3fc'

    icl_model = impl_models.ICL_DenseNet_3fc
    path_to_model_weights = '../centrioles/models/ICL_DenseNet_3fc/true_save/weights/'
    trainer = Trainer(model)
    if torch.cuda.is_available():
        trainer = trainer.load(from_directory=path_to_model_weights,
                               best=True)
    else:
        trainer = trainer.load(from_directory=path_to_model_weights,
                               best=True, map_location='cpu')
    icl_model = trainer.model
    icl_model.features_needed = True

    init_weights(model, icl_model, lambda x: 'main_blocks' in x)

    # DIRS
    model_dir = os.path.join('models', model_name)
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
        .validate_every((2, 'epochs')) \
        .save_every((5, 'epochs')) \
        .save_to_directory(weight_dir) \
        .set_max_num_epochs(10000) \
        .build_logger(logger, log_directory=logs_dir) \
        .register_callback(AutoLR(0.9, (1, 'epochs'),
                           consider_improvement_with_respect_to='previous'))

    # Bind loaders
    trainer \
        .bind_loader('train', train_dl) \
        .bind_loader('validate', test_dl)

    if torch.cuda.is_available():
        trainer.cuda()

    trainer.fit()