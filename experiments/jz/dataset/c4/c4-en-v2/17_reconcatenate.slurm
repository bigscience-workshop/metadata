#!/bin/bash
#SBATCH --job-name=concatenate-ds-mod-meta-bs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40         # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=cpu_p1
#SBATCH --time 01:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/concatenate-arrow/%j-%x.out           # output file name
#SBATCH --array=0
#SBATCH --account=six@cpu
#SBATCH --qos=qos_cpu-dev

# ===== GET DATASET ======

set -x -e

source $HOME/start-modelling-metadata-user


LIST_DATASETS_FILE=$WORK/repos/metadata/experiments/jz/dataset/c4/c4-en-v2/configs/17_config.txt

DATASET_ID=$SLURM_ARRAY_TASK_ID
LIST_DATASET=($(cat $LIST_DATASETS_FILE))
DATASET_NAME=${LIST_DATASET[$SLURM_ARRAY_TASK_ID]}
echo $DATASET_NAME

ARROW_DATASET_SHARDED_WITH_ENTITIES_DIR="/gpfsscratch/rech/six/uue59kq/new_dataset/process-v2/c4-en-sharded-with-entity/"
ARROW_DATASET_WITH_ENTITIES_DIR="/gpfsscratch/rech/six/uue59kq/new_dataset/process-v2/c4-en-concatenate-with-entity/"

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=$SCRATCH/to_delete

MODELING_METADATA_REPO=$WORK/repos/metadata/

cd $MODELING_METADATA_REPO

python experiments/jz/dataset/c4/python_scripts/concatenate_dataset.py \
    --dir-dataset-path $ARROW_DATASET_SHARDED_WITH_ENTITIES_DIR \
    --dataset-name $DATASET_NAME \
    --save-path $ARROW_DATASET_WITH_ENTITIES_DIR\
    --number_shards 10