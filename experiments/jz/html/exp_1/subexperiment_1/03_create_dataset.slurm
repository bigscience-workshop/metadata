#!/bin/bash
#SBATCH --job-name=modelling-metadata-html-metadata-exp1-subexp1-create-dataset     # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                            # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=20                                             # (change me! between 0 and 40) number of cores per tasks
#SBATCH --hint=nomultithread                                           # we get physical cores not logical
#SBATCH --time 02:00:00                                                # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
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
source $HOME/start-user

# We are on an offline partition
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Folder for the clone of github.com/bigscience-workshop/metadata/
cd $WORK/repos/sync/metadata/

HUB_REPO_NAME='SaulLu/Natural_Questions_HTML_reduced_all' # (change me! e.g. SaulLu/Natural_Questions_HTML_Toy_v2)

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

THEPATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
THIS_SCRIPT_DIR=$( dirname $THEPATH)
echo "$THIS_SCRIPT_DIR"

# Now we launch the script that will perform the preprocessing of the dataset
# Feel free to add any arguments you like (change me!)
python experiments/jz/html/exp_1/start_training.py \
  --config-dir=$THIS_SCRIPT_DIR \
  data_config.dataset_name="${REPO_DIR}" \
  data_config.overwrite_cache=true \
  out_dir="${SCRATCH}/metadata_outputs" \
  jobid="${SLURM_JOB_ID}" \
  do_train=false \
  do_eval=false
