#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-dataset-create-toy             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 02:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out   # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err    # error file name
#SBATCH --account=six@gpu                                               # account
#SBATCH -p compil                                                       # partition with internet

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have only two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-user

cd $SCRATCH/new_dataset/

DIR_NAME_TMP=c4-mike-en-tmp
DIR_NAME=c4-mike-en
NEW_DATASET_DIR=$SCRATCH/new_dataset/$DIR_NAME

if [[ -d "${DIR_NAME_TMP}" ]]; then
    echo "${DIR_NAME_TMP} already exists on your filesystem."
else
    echo "${DIR_NAME_TMP} doesn't exists on your filesystem."
    mkdir $DIR_NAME_TMP
fi

if [[ -d "${DIR_NAME}" ]]; then
    echo "${DIR_NAME} already exists on your filesystem."
else
    echo "${DIR_NAME} doesn't exists on your filesystem."
    mkdir $DIR_NAME
fi

if [[ -d "${NEW_DATASET_DIR}" ]]; then
    echo "${NEW_DATASET_DIR} already exists on your filesystem."
else
    echo "${NEW_DATASET_DIR} doesn't exists on your filesystem."
    mkdir $NEW_DATASET_DIR
fi

cd $DIR_NAME_TMP
git clone https://huggingface.co/datasets/bs-modeling-metadata/openwebtext-html-cc

cd openwebtext-html-cc
git lfs pull --include "*c4-en-html/*"

cp $DIR_NAME_TMP/openwebtext-html-cc/c4-en-html $NEW_DATASET_DIR
