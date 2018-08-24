#!/g/kreshuk/lukoianov/miniconda3/envs/inferno/bin/python3

#DEFAULT SETTINGS
PROJECT    = '_centrioles_detection'
GROUP_NAME = 'kreshuk'
EMAIL      = 'artem.lukoianov@embl.de'
MEMORY     = 20
TIME_LIMIT = 10
PREFIX     = ''
MAIL_TYPE  = 'FAIL'

ADDITIONAL_MODULES = 'module load cuDNN'
ARGUMENTS_FOR_RUN  = ''

slurm_script_template = \
'''#!/bin/bash

#SBATCH -J {}_{}{}
#SBATCH -A {}
#SBATCH -N 1
#SBATCH -n 3
#SBATCH --mem {}G
#SBATCH -t {}:00:00
#SBATCH -o {}/outfile.log
#SBATCH -e {}/errfile.log
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
    parser.add_argument('--time', type=int, default=TIME_LIMIT, dest='time',
                        help='Time limit for the script execution')

    args, unknown = parser.parse_known_args()

    kargs = ARGUMENTS_FOR_RUN + '--id ' + args.prefix + ' ' + ' '.join(unknown)

    parent_dir = 'models/{}/'.format(args.arch + '_' + args.prefix)
    if os.path.exists(parent_dir):
        print('Directory already exists! Aborting')
        exit()
    os.makedirs(parent_dir)

    bash_script_text = slurm_script_template.format(args.prefix, args.arch, PROJECT, GROUP_NAME, args.mem, args.time, 
                                                    parent_dir, parent_dir, MAIL_TYPE, EMAIL) + '\n' +\
                        ADDITIONAL_MODULES + '\n' + RUNNING_COMANDS[args.arch] + ' ' + kargs
                                    
    with open('slurm_script.sh', 'w') as f:
        print(bash_script_text, file=f)

    os.system('rm -rf {}logs'.format(parent_dir))
    os.system('sbatch slurm_script.sh')