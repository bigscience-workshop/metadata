#!/bin/bash
#SBATCH --job-name=modelling-metadata-website-desc-load-model-and-tokenizer-25k  # job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                                      # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                                        # number of cores per tasks
#SBATCH --hint=nomultithread                                                     # we get physical cores not logical
#SBATCH --gres=gpu:0                                                             # number of gpus
#SBATCH --time 00:30:00                                                          # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out            # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.err             # error file name
#SBATCH --account=six@gpu                                                        # account
#SBATCH -p compil                                                                # partition with internet

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-modelling-metadata-user

# Folder for the clone of github.com/bigscience-workshop/metadata/
cd $WORK/repos/metadata/

# Command to load the XXX model and tokenizer stored on https://huggingface.co/models
python experiments/jz/utils/loading_script_utils/load_tokenizer_and_model.py \
    model_name=gpt2 