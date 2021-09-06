#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-do-train-test # job name
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --constraint=v100-16g
#SBATCH --gres=gpu:1                  # number of GPUs per node
#SBATCH --cpus-per-task=8             # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time 00:10:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out            # output file name
#SBATCH --error=%x-%j.err             # error file name
#SBATCH --account=six@gpu             # account

set -x -e

source $HOME/start-user

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline

cd $WORK/repos/test-sync/metadata/

python experiments/html/start_training.py \
data_config.experiment="with_metadata" \
data_config.metadata_list=["html"] \
data_config.max_seq_len=1024 \
data_config.dataset_name="${DATASETS_CUSTOM}/Natural_Questions_HTML_Toy"  \
data_config.train_file="nq-train-*.jsonl.gz" \
data_config.validation_file="nq-dev-*.jsonl.gz" \
data_config.extension="json" \
data_config.preprocessing_num_workers=6 \
out_dir="${SCRATCH}/metadata_outputs" \
do_train=True \
do_eval=True \