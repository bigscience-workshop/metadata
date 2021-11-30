#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-dataset-create-toy             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --time 00:10:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
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

DIR_NAME_TMP=c4-en-reduced-tmp
if [[ -d "${DIR_NAME_TMP}" ]]; then
    echo "${DIR_NAME_TMP} already exists on your filesystem."
else
    echo "${DIR_NAME_TMP} doesn't exists on your filesystem."
    mkdir $DIR_NAME_TMP
fi

DIR_NAME=c4-en-reduced
if [[ -d "${DIR_NAME}" ]]; then
    echo "${DIR_NAME} already exists on your filesystem."
else
    echo "${DIR_NAME} doesn't exists on your filesystem."
    mkdir $DIR_NAME
fi

cp $DSDIR/c4/en/c4-train.00000-of-01024.json.gz $SCRATCH/new_dataset/c4-en-reduced-tmp
cp $DSDIR/c4/en/c4-train.00001-of-01024.json.gz $SCRATCH/new_dataset/c4-en-reduced-tmp

cd $WORK/repos/sync/metadata/

NEW_DATASET_DIR=$SCRATCH/new_dataset/c4-en-reduced-1000
python experiments/jz/dataset/c4/python_scripts/dataset_reducer.py \
    --data-path $SCRATCH/new_dataset/c4-en-reduced-tmp \
    --new-dataset-path $NEW_DATASET_DIR \
    --total-number-examples 1000

cd $NEW_DATASET_DIR
rename -v 'c4-train.' 'c4-train_' *