#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-create-dataset-test # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8            # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time 00:10:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out          # output file name
#SBATCH --error=%x-%j.err           # error file name
#SBATCH --account=six@cpu # account

set -x -e

source $HOME/start-user

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $WORK/repos/metadata/

python experiments/html/start_training.py \
data_config.experiment="with_metadata" \
data_config.metadata_list=["html"] \
data_config.max_seq_len=1024 \
data_config.dataset_name="${DATASETS_CUSTOM}/SaulLu/Natural_Questions_HTML_Toy" \
data_config.train_file="nq-train-*.jsonl.gz" \
data_config.validation_file="nq-dev-*.jsonl.gz" \
data_config.extension="json" \
data_config.preprocessing_num_workers=8 \
do_train=False \
do_eval=False \