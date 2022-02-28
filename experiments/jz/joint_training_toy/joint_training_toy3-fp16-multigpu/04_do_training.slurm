#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-joint_training_toy3-fp16-multigpu
#SBATCH --ntasks=1                    # number of MP tasks
#SBATCH --constraint=v100-16g
#SBATCH --cpus-per-task=8                                              # (change me! between 0 and 40) number of cores per tasks
#SBATCH --hint=nomultithread                                           # we get physical cores not logical
#SBATCH --time 01:00:00                                                # (change me! between 0 and 20h) maximum execution time (HH:MM:SS) 
#SBATCH --gres=gpu:2                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out  # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err   # error file name         # error file name
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

HUB_REPO_NAME='bs-modeling-metadata/c4-en-reduced-with-metadata' # (change me! e.g. SaulLu/Natural_Questions_HTML_Toy_v2)

# We define the name of the folder in which the clone will be made
#Define multi-character delimiter
delimiter="/"
#Concatenate the delimiter with the main string
string=$HUB_REPO_NAME$delimiter

#Split the text based on the delimiter
myarray=()
while [[ $string ]]; do
  myarray+=( "${string%%"$delimiter"*}" )
  string=${string#*"$delimiter"}
done
REPO_DIR="${DATASETS_CUSTOM}/${myarray[-1]}"

echo "compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 2
" > accelerate_config.yaml

accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py \
    data_config.experiment="with_metadata" \
    data_config.metadata_config.metadata_list='[entity, timestamp, url, website_description]' \
    data_config.metadata_config.max_seq_len=1024 \
    data_config.dataset_name="${REPO_DIR}" \
    data_config.preprocessing_num_workers=8 \
    out_dir="${SCRATCH}/metadata_outputs/${SLURM_JOB_ID}" \
    jobid="${SLURM_JOB_ID}" \
    do_train=True \
    do_eval=True \
    evaluation_strategy=STEPS \
    eval_steps=50 \
    save_strategy=STEPS \
    save_steps=50 \
    gradient_accumulation_steps=2
