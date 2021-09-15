#!/bin/bash
#SBATCH --job-name=modelling-metadata-example-load-dataset              # (change me!) job name
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
source $HOME/start-modelling-metadata-user

# For the moment we can't directly use the new dataset feature on JZ which would avoid having to clone the dataset
# repo from the HUB. So the first thing to do is to clone the repo of the XXX dataset if it does not already exist.
HUB_REPO_NAME='SaulLu/Natural_Questions_HTML_Toy' # (change me! e.g. SaulLu/Natural_Questions_HTML_Toy)

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

# We clone the repo if it doesn't exist
if [[ -d "${REPO_DIR}" ]]; then
    echo "${REPO_DIR} already exists on your filesystem."
else
    echo "${REPO_DIR} doesn't exists on your filesystem."
    cd $DATASETS_CUSTOM/
    git clone "https://huggingface.co/datasets/${HUB_REPO_NAME}"
    cd ${REPO_DIR}
    git lfs install
    git lfs pull origin master
fi

cd $WORK/repos/sync/metadata/

# We check that the dataset can indeed be loaded
python experiments/jz/utils/loading_script_utils/load_dataset.py \
    dataset_name="${REPO_DIR}" \
    train_file="nq-train-*.jsonl.gz" \
    validation_file="nq-dev-*.jsonl.gz"
