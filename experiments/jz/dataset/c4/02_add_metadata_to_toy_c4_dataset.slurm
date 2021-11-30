#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-dataset-toy-add-metadata             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 01:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/create-dataset-toy/%j-%x.out   # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/create-dataset-toy/%j-%x.err    # error file name
#SBATCH --account=six@cpu                                               # account

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have only two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-modelling-metadata-user

cd $WORK/repos/sync/metadata/

export FLAIR_CACHE_ROOT=$SCRATCH/cache_dir/flair

echo "Args:"
echo "    file_name=$FILENAME"
echo "    out_file_name=$NEW_FILENAME"
echo "    out_dir=$OUT_DIR"
echo "    dataset_name=$DATASET_FILES_DIR"
echo "    metadata_to_include=$METADATA_TO_INCLUDE"

echo "Processing $DATASET_FILES_DIR/$FILENAME to add timestamp and save results into $OUT_DIR/$NEW_FILENAME"
python experiments/jz/dataset/c4/python_scripts/add_metadata.py \
    file_name=$FILENAME \
    out_file_name=$NEW_FILENAME \
    out_dir=$OUT_DIR \
    website_desc_path_wiki_db=$SCRATCH/modeling-metadata-artefacts/wiki_en_dump.db \
    entity_path_data_dir=$SCRATCH/modeling-metadata-artefacts/entity_preprocessing \
    dataset_name=$DATASET_FILES_DIR \
    metadata_to_include=$METADATA_TO_INCLUDE \
    path_or_url_flair_ner_model=$PATH_OR_URL_FLAIR_NER_MODEL

gzip $OUT_DIR/$NEW_FILENAME

echo "$OUT_DIR/$NEW_FILENAME created"