#!/bin/bash

../rsub run_densenet.py
../rsub run_simple.py
../rsub run_mil.py
../rsub src/datasets.py
../rsub src/utils.py
../rsub src/architectures.py

export LC_ALL="en_US.UTF-8"
export LC_CTYPE="en_US.UTF-8"

echo
echo To run tensorboard:
echo tensorboard --logdir=models --port=6007
echo
echo To submit densnet:
echo ./submit_job.py --arch densenet --id 01 --BN --GR 5 --red 0.5 --depth 46
