#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-dataset-gzip-data             # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=1                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 08:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/create-dataset-c4-toy/%j-%x.out   # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/create-dataset-c4-toy/%j-%x.err    # error file name
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=0%1
#SBATCH --partition=cpu_p1

set -x -e

PROCESS_DIR=$SCRATCH/new_dataset/c4-mike-en-test-5
OUT_DIR=$PROCESS_DIR/c4-en-reduced-with-metadata-url-html-timestamp
cd $OUT_DIR

gzip c4-*${SLURM_ARRAY_TASK_ID}-*.jsonl