#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#DEFAULT SETTINGS
PROJECT    = 'centrioles_detection'
GROUP_NAME = 'kreshuk'
EMAIL      = 'artem.lukoianov@embl.de'
MEMORY     = 20
TIME_LIMIT = 10
PREFIX     = ''
MAIL_TYPE  = 'FAIL'

ADDITIONAL_MODULES = 'module load cuDNN'
RUNNING_COMANDS    = {'densenet' : './run_densenet.py',
                      'mil'      : './run_mil.py',
                      'simple'   : './run_simple.py'}
ARGUMENTS_FOR_RUN  = ''

slurm_script_template = \
'''#!/bin/bash

#SBATCH -J {}{}
#SBATCH -A {}
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem {}G
#SBATCH -t {}:00:00
#SBATCH -o {}outfile.log
#SBATCH -e {}errfile.log
#SBATCH --mail-type={}
#SBATCH --mail-user={}
#SBATCH -p gpu
#SBATCH -C gpu=1080Ti
#SBATCH --gres=gpu:1
'''

import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Submit job to the cluster')
    parser.add_argument('--id', metavar='pref', type=str, default=PREFIX, dest='prefix',
                        help='Prefix for the ouput files')
    parser.add_argument('--mem', type=int, default=MEMORY, dest='mem',
                        help='Amount of RAM to be reserved')
    parser.add_argument('--arch', action='store', choices=RUNNING_COMANDS.keys(),
                      help='Architecture of network to run')
    parser.add_argument('--time', type=int, default=TIME_LIMIT, dest='time',
                        help='Time limit for the script execution')

    args = parser.parse_args()

    if args.prefix != '':
        args.prefix += '_'

    bash_script_text = slurm_script_template.format(args.prefix, PROJECT, GROUP_NAME, args.mem, args.time, 
                                                    args.prefix, args.prefix, MAIL_TYPE, EMAIL) + '\n' +\
                        ADDITIONAL_MODULES + '\n' + RUNNING_COMANDS[args.arch] + ' ' + ARGUMENTS_FOR_RUN
                                    
    with open('slurm_script.sh', 'w') as f:
        print(bash_script_text, file=f)

    os.system('sbatch slurm_script.sh')