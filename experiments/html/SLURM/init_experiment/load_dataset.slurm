#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-create-dataset-test # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --time 00:02:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --error=%x-%j.err            # error file name
#SBATCH --account=six@cpu            # account
#SBATCH -p compil                   # partition with internet

set -x -e

source $HOME/start-user

cd $WORK/repos/metadata/

python experiments/html/SLURM/init_experiment/load_dataset.py \
dataset_name="SaulLu/Natural_Questions_HTML_Toy" \
train_file="nq-train-*.jsonl.gz" \
validation_file="nq-dev-*.jsonl.gz" 