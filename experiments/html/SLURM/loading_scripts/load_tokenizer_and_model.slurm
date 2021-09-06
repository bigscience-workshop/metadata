#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-download-tokenizer-and-model # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --time 00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err            # error file name
#SBATCH --account=six@gpu            # account
#SBATCH -p compil                    # partition with internet

set -x -e

source $HOME/start-user

cd $WORK/repos/test-sync/metadata/

python experiments/html/SLURM/loading_scripts/load_tokenizer_and_model.py \
model_name=gpt2 \