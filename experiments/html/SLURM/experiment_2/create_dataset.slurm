#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-create-dataset-test # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40            # number of cores per tasks
#SBATCH --hint=nomultithread          # we get physical cores not logical
#SBATCH --time 01:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out          # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err           # error file name
#SBATCH --account=six@cpu # account

set -x -e

source $HOME/start-user

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $WORK/repos/sync/metadata/

python experiments/html/start_training.py \
data_config.experiment="with_metadata_and_baseline_val" \
data_config.metadata_list=["html"] \
data_config.max_seq_len=1024 \
data_config.dataset_name="${DATASETS_CUSTOM}/Natural_Questions_HTML" \
data_config.train_file="nq-train-0\[0-2\].jsonl.gz" \
data_config.validation_file="nq-dev-00.jsonl.gz" \
data_config.extension="json" \
data_config.preprocessing_num_workers=80 \
out_dir="${SCRATCH}/metadata_outputs" \
do_train=False \
do_eval=False \