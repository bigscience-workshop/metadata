#!/bin/bash
#SBATCH --job-name=shard-ds-mod-meta-bs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/shard-arrow/%j-%x.out           # output file name
#SBATCH --array=6-256
#SBATCH --account=six@cpu

# ===== GET DATASET ======

set -x -e

source $HOME/start-modelling-metadata-user


LIST_DATASETS_FILE="/gpfsssd/scratch/rech/six/uue59kq/tmp/listfiles.txt" # Change me

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=($(cat $LIST_DATASETS_FILE))
DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]%.jsonl.gz*}
echo $DATASET_NAME

ARROW_DATASET_DIR="/gpfsscratch/rech/six/uue59kq/new_dataset/c4-en-html-with-metadata-arrow/"$DATASET_NAME
ARROW_DATASET_SHARDED_DIR="/gpfsscratch/rech/six/uue59kq/new_dataset/c4-en-html-with-metadata-sharded/"

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

MODELING_METADATA_REPO=$WORK/repos/metadata/

cd $MODELING_METADATA_REPO

python experiments/jz/dataset/c4/python_scripts/shard_ds.py \
    --dataset-path $ARROW_DATASET_DIR \
    --save-path $ARROW_DATASET_SHARDED_DIR \
    --number_shards 10 \