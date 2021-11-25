#!/bin/bash
#SBATCH --job-name=modelling-metadata-website-desc-clean-train-25k     # job name
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --constraint=v100-16g
#SBATCH --gres=gpu:1                                                   # number of GPUs per node
#SBATCH --cpus-per-task=8                                              # (change me! between 0 and 40) number of cores per tasks
#SBATCH --hint=nomultithread                                           # we get physical cores not logical
#SBATCH --time 20:00:00                                                # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out  # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err   # error file name         
#SBATCH --account=six@gpu                                              # account

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-modelling-metadata-user

# We are on an offline partition
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH

# Folder for the clone of github.com/bigscience-workshop/metadata/
cd $WORK/repos/metadata/

HUB_REPO_NAME='bs-modeling-metadata/website_metadata_c4'

# We define the name of the folder in which the clone will be made
#Define multi-character delimiter
delimiter="/"
#Concatenate the delimiter with the main string
string=$HUB_REPO_NAME$delimiter

#Split the text based on the delimiter
myarray=()
while [[ $string ]]; do
  myarray+=("${string%%"$delimiter"*}")
  string=${string#*"$delimiter"}
done
REPO_DIR="${DATASETS_CUSTOM}/${myarray[-1]}"

python bsmetadata/train.py \
  data_config.experiment="with_metadata" \
  data_config.metadata_config.metadata_list=["website_description"] \
  data_config.dataset_name="${REPO_DIR}" \
  data_config.train_file="website_metadata_clean_train.jsonl.gz" \
  data_config.validation_file="website_metadata_clean_dev.jsonl.gz" \
  data_config.preprocessing_num_workers=80 \
  data_config.per_device_eval_batch_size=3 \
  data_config.per_device_train_batch_size=3 \
  out_dir="${SCRATCH}/metadata_outputs/${SLURM_JOB_ID}" \
  jobid="${SLURM_JOB_ID}" \
  do_train=True \
  do_eval=True \
  num_train_epochs=3 \
  evaluation_strategy=EPOCH \
  save_strategy=EPOCH \
  save_num_per_epoch=1 \
  eval_num_per_epoch=1