#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-download-dataset-test # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --gres=gpu:0                 # number of gpus
#SBATCH --time 02:30:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out           # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err            # error file name
#SBATCH --account=six@gpu            # account
#SBATCH -p compil                    # partition with internet

set -x -e

source $HOME/start-user

# Uncomment if the repo doesn't exist
# cd $DATASETS_CUSTOM/
# git clone https://huggingface.co/datasets/SaulLu/Natural_Questions_HTML
# cd Natural_Questions_HTML_Toy/
# git lfs install
# git lfs pull origin master

cd $WORK/repos/test-sync/metadata/

python experiments/html/SLURM/init_experiment/load_dataset.py \
dataset_name="${DATASETS_CUSTOM}/Natural_Questions_HTML" \
train_file="nq-train-*.jsonl.gz" \
validation_file="nq-dev-*.jsonl.gz" 