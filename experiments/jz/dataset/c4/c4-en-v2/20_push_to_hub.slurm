#!/bin/bash
#SBATCH --job-name=push-to-hub-mod-meta-bs              # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=40                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 20:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --gres=gpu:0                                                    # (change me! between 0 and 1) number of gpus
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/push-to-hub/%j-%x.out           # output file name
#SBATCH --account=six@gpu                                               # account
#SBATCH -p compil                                                       # partition with internet

set -x -e

source $HOME/start-modelling-metadata-user

cd $SCRATCH/new_dataset/c4-en-html-with-metadata

git status

# start_time=$SECONDS
# git add -v *.gz
# elapsed=$(( SECONDS - start_time ))
# echo "Git add took: "$elapsed" s"

# start_time=$SECONDS
# git add -v dataset_entities_info.json
# elapsed=$(( SECONDS - start_time ))
# echo "Git add took: "$elapsed" s"

# start_time=$SECONDS
# git commit -v -m "add dataset with entities"
# elapsed=$(( SECONDS - start_time ))
# echo "Git commit took: "$elapsed" s"

start_time=$SECONDS
git push
elapsed=$(( SECONDS - start_time ))
echo "Git push took: "$elapsed" s"