JID_JOB3=$(sbatch 03_create_dataset.slurm | cut -d " " -f 4)
sbatch --dependency=afterok:$JID_JOB3 04_do_training.slurm
