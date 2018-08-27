#!/bin/bash

../rsub run_model.py
../rsub src/datasets.py
../rsub src/utils.py
../rsub src/architectures.py
../rsub src/implemented_models.py

echo
echo To run tensorboard:
echo source activate inferno
echo export LC_ALL="en_US.UTF-8"
echo export LC_CTYPE="en_US.UTF-8"
echo tensorboard --logdir=models --port=6007
echo
echo To submit model:
echo ./submit_job.py --model_name CNN_512_7conv_to4x4_3fc --lr 0.001 --id 01 
