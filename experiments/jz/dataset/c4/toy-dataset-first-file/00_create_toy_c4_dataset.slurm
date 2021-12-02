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

DIR_NAME=$SCRATCH/new_dataset/c4-en-reduced-first-file

mkdir $DIR_NAME

cp $DSDIR/c4/en/c4-train.00000-of-01024.json.gz $DIR_NAME/c4-train.00000-of-01024.json.gz

cd $DIR_NAME
rename -v 'c4-train.' 'c4-train_' *