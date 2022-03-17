#!/bin/bash
#SBATCH --job-name=add-entities-mod-meta-bs         # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 08:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/add-entities-test/%j-%x.out  
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=0-5
#SBATCH --partition=cpu_p1

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have only two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-modelling-metadata-user

cd $WORK/repos/metadata/

# We are on an offline partition
export FLAIR_CACHE_ROOT=$SCRATCH/cache_dir/flair
export HF_DATASETS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH
export HF_DATASETS_CACHE=$SCRATCH/to_delete

METADATA_TO_INCLUDE='["entity"]'

PROCESS_DIR=/gpfsscratch/rech/six/uue59kq/new_dataset
DATASET_FILES_DIR=$PROCESS_DIR/c4-en-html-with-metadata-arrow
OUT_DIR=$PROCESS_DIR/process-v2/c4-en-reduced-with-entity

mkdir -p $OUT_DIR

MAP_BATCH_SIZE=100
PREPROCESSING_NUM_WORKERS=1
NUM_FILES_TO_PROCESS=1
SAVE_BATCH_SIZE=100

PATH_OR_URL_FLAIR_NER_MODEL=$SCRATCH/cache_dir/flair/ner-fast/en-ner-fast-conll03-v0.4.pt

echo "Args:"
echo "    task_id=${SLURM_ARRAY_TASK_ID}"
echo "    map_batch_size=$MAP_BATCH_SIZE"
echo "    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS"
echo "    out_dir=$OUT_DIR"
echo "    dataset_name=$DATASET_FILES_DIR"
echo "    metadata_to_include=$METADATA_TO_INCLUDE"
echo "    path_or_url_flair_ner_model=$PATH_OR_URL_FLAIR_NER_MODEL"

python experiments/jz/dataset/c4/python_scripts/add_metadata.py \
    task_id=${SLURM_ARRAY_TASK_ID}\
    out_dir=$OUT_DIR \
    path_wiki_db=$SCRATCH/modeling-metadata-artefacts/wiki_en_dump.db \
    entity_path_data_dir=$SCRATCH/modeling-metadata-artefacts/entity_preprocessing \
    dataset_name=$DATASET_FILES_DIR \
    metadata_to_include="$METADATA_TO_INCLUDE" \
    path_or_url_flair_ner_model=$PATH_OR_URL_FLAIR_NER_MODEL \
    map_batch_size=$MAP_BATCH_SIZE \
    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS\
    num_files_to_process=$NUM_FILES_TO_PROCESS\
    save_batch_size=$SAVE_BATCH_SIZE\

