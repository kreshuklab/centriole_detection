#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#BASIC IMPORTS
import argparse
import os
import subprocess
import sys
import dill
 
#INTERNAL IMPORTS
from src.datasets import CentriollesDatasetBags, GENdataset
from src.utils import get_basic_transforms, log_info, init_weights
from src.architectures import DenseNet
from src.implemented_models import ICL_DenseNet_3fc, ICL_MIL_DS3fc

#INFERNO IMPORTS
import torch
from inferno.trainers.basic import Trainer
from torch.utils.data import DataLoader
from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of simple CNN implementation')

    parser.add_argument('--model_name', type=str, default='', help='Name of the model from models dir')
    parser.add_argument('--arti', action='store_true', help='Teach on the artificial data')
    parser.add_argument('--id', type=str, default='default', help='Unique net id to save')
    parser.add_argument('--img_size', type=int, default=512, help='Size of input images')
    parser.add_argument('--stride', type=float, default=0.5, help='From 0 to 1')
    parser.add_argument('--wsize', type=int, default=60, help='Size of windows for bagging')

    args = parser.parse_args()
    log_info( 'Params: ' + str(args))

    train_tr, test_tr = get_basic_transforms()
    if args.arti:
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
    
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=1, shuffle=True, num_workers=0)

    log_info('Datasets are initialized!')

    ## INIT model
    def load(self, from_directory=None, best=False, filename=None, map_location=None):
        from_directory = self._save_to_directory if from_directory is None else from_directory
        assert from_directory is not None, "Nowhere to load from."
        # Get file name
        if filename is None:
            filename = self._best_checkpoint_filename if best else self._checkpoint_filename
        # Load the dictionary
        config_dict = torch.load(os.path.join(from_directory, filename),
                                pickle_module=dill, map_location=map_location)

        # This is required to prevent an infinite save loop?
        self._is_iteration_with_best_validation_score = False
        # Set config
        self.set_config(config_dict)
        return self

    Trainer.load = load

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
        .build_optimizer('Adam') \
        .validate_every((2, 'epochs')) \
        .save_every((5, 'epochs')) \
        .save_to_directory(model_dir) \
        .set_max_num_epochs(10000) \
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