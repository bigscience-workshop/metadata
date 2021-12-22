#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-joint-toy-1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                            # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20                                             # (change me! between 0 and 40) number of cores per tasks
#SBATCH --hint=nomultithread                                           # we get physical cores not logical
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --time 01:00:00                                                # (change me! between 0 and 20h) maximum execution time (HH:MM:SS) 
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out  # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err   # error file name
#SBATCH --account=six@cpu # account

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

# Now we launch the script that will perform the preprocessing of the dataset
# Feel free to add any arguments you like (change me!)
REPO_DIR='bs-modeling-metadata/c4-en-reduced-with-metadata'
out_dir="${SCRATCH}/metadata_outputs"
python bsmetadata/train.py \
    data_config.experiment="with_metadata" \
    data_config.metadata_config.metadata_list='[entity, timestamp, url, website_description]' \
    data_config.metadata_config.max_seq_len=1024 \
    data_config.dataset_name="${REPO_DIR}" \
    data_config.preprocessing_num_workers=8 \
    out_dir="${SCRATCH}/metadata_outputs" \
    do_train=False \
    do_eval=False
