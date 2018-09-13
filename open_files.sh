#!/bin/bash

../rsub run_model.py
../rsub src/datasets.py
../rsub src/utils.py
../rsub src/architectures.py
../rsub src/implemented_models.py
../rsub src/trainer.py

echo
echo To run tensorboard:
echo source activate inferno
echo export LC_ALL="en_US.UTF-8"
echo export LC_CTYPE="en_US.UTF-8"
echo tensorboard --logdir=/g/kreshuk/lukoianov/centrioles/models --port=6007
echo
echo To submit model:
echo ./submit_job.py --model_name MIL_32x32_to4x4 --lr 0.00001 --id right_norm --use_bags --epoch 1000 --save_best --save_each 50
