#!/bin/bash
#SBATCH --job-name=modelling-metadata-website-desc-create-dataset-25k  # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                            # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20                                             # number of cores per tasks
#SBATCH --hint=nomultithread                                           # we get physical cores not logical
#SBATCH --time 01:00:00                                                # maximum execution time (HH:MM:SS)
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

# Folder for the clone of github.com/bigscience-workshop/metadata/
cd $WORK/repos/sync/metadata/

# Now we launch the script that will perform the preprocessing of the dataset
# Feel free to add any arguments you like (change me!)
python experiments/jz/toy_experiments/dumpdb.py \
    --path-to-dump "${SCRATCH}/tmp/en_wiki_new.db"