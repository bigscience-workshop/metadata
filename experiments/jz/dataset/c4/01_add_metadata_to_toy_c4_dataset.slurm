#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-dataset-toy-add-metadata             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --time 01:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
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

cd $WORK/repos/sync/metadata/

DATASET_FILES_DIR=$SCRATCH/new_dataset/hub/c4-en-reduced

for filename in $DATASET_FILES_DIR/*; do
    FILENAME="${filename##*/}"
    NEW_FILENAME="${FILENAME::-3}l" #   the last 3 characters correspond to the compression algorithm format .gz
    echo "Processing $FILENAME and save results into $NEW_FILENAME"
    python experiments/jz/dataset/c4/python_scripts/add_metadata.py \
        file_name=$FILENAME \
        out_file_name=$NEW_FILENAME \
        out_dir=$SCRATCH/new_dataset/c4-en-reduced-with-metadata \
        website_desc_path_wiki_db=$SCRATCH/modeling-metadata-artefacts/wiki_en_dump.db \
        entity_path_data_dir=$SCRATCH/modeling-metadata-artefacts/entity_preprocessing \
        dataset_name=$DATASET_FILES_DIR
        
    gzip $SCRATCH/new_dataset/c4-en-reduced-with-metadata/$NEW_FILENAME
    echo "Processing $FILENAME finished"
done
