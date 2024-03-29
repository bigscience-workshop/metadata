#!/bin/bash
#SBATCH --job-name=export-to-jsonlines-mod-meta-bs          # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 20:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)             # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/to-jsonlies/%j-%x.out           # output file name
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=0-256
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


PROCESS_DIR=/gpfsscratch/rech/six/uue59kq/new_dataset/process-v2/
DATASET_FILES_DIR=$PROCESS_DIR/c4-en-concatenate-with-entity
OUT_DIR=$PROCESS_DIR/c4-en-concatenate-with-entity-jsonl-compressed

mkdir -p $OUT_DIR

PREPROCESSING_NUM_WORKERS=1
NUM_FILES_TO_PROCESS=1
SAVE_BATCH_SIZE=1000

echo "Args:"
echo "    task_id=${SLURM_ARRAY_TASK_ID}"
echo "    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS"
echo "    out_dir=$OUT_DIR"
echo "    dataset_name=$DATASET_FILES_DIR"

python experiments/jz/dataset/c4/python_scripts/export_to_compressed_jsonl.py \
    task_id=${SLURM_ARRAY_TASK_ID}\
    out_dir=$OUT_DIR \
    dataset_name=$DATASET_FILES_DIR \
    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS\
    num_files_to_process=$NUM_FILES_TO_PROCESS\
    save_batch_size=$SAVE_BATCH_SIZE\
    use_load_from_disk=true
