#!/bin/bash
#SBATCH --job-name=modelling-metadata-sync-wandb  # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --nodes=1                    # number of nodes
#SBATCH --cpus-per-task=1            # number of cores per task
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=2:00:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/%x-%j.out           # output file name
#SBATCH --partition=compil
#SBATCH --account=six@cpu

echo "START TIME: $(date)"

source $HOME/start-user

cd ${SCRATCH}

while true
do 
    wandb sync --sync-all
    sleep 30
done

echo "END TIME: $(date)"