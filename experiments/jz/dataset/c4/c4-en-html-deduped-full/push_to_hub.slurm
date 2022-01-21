#!/bin/bash
#SBATCH --job-name=modelling-metadata-c4-push-to-hub               # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=8                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 10:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/c4-en-html-deduped-push-to-hub/%x-%j.out   # output file name
#SBATCH --error=/gpfsdswork/projects/rech/six/uue59kq/logs/c4-en-html-deduped-push-to-hub/%x-%j.err    # error file name
#SBATCH --account=six@gpu                                               # account
#SBATCH -p compil                                                       # partition with internet

set -x -e

source $HOME/start-modelling-metadata-user

cd $SCRATCH/new_dataset/c4-en-html-deduped-processed/c4-en-html-with-metadata

git status

git add -v *.gz

git add -v dataset_info.json

git commit -v -m "add dataset with html, timestamp, url, datasource, generation length, website description, HTML footers, HTML headers, HTML titles"

git push