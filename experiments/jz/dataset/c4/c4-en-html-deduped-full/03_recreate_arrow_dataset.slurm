#!/bin/bash
#SBATCH --job-name=recreate-arrow-ds-mod-meta-bs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/recreate-arrow/%j-%x.out           # output file name
#SBATCH --array=0-2756
#SBATCH --account=six@cpu

# ===== GET DATASET ======

set -x -e

source $HOME/start-modelling-metadata-user


LIST_DATASETS_FILE="/gpfsssd/scratch/rech/six/uue59kq/tmp/listfiles.txt" # Change me

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=($(cat $LIST_DATASETS_FILE))
DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]%.jsonl.gz*}
echo $DATASET_NAME

JSON_DATASET_FILES="/gpfsscratch/rech/six/uue59kq/new_dataset/c4-en-html-with-metadata/"$DATASET_NAME.jsonl.gz
ARROW_DATASET_DIR="/gpfsscratch/rech/six/uue59kq/new_dataset/c4-en-html-with-metadata-arrow/"$DATASET_NAME

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

MODELING_METADATA_REPO=$WORK/repos/metadata/

cd $MODELING_METADATA_REPO

python experiments/jz/dataset/c4/python_scripts/create_arrow_dataset.py \
    --dataset-path $JSON_DATASET_FILES \
    --save-path $ARROW_DATASET_DIR