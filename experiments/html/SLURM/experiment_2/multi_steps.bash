JID_JOB1=`sbatch  create_dataset.slurm | cut -d " " -f 4`
sbatch  --dependency=afterok:$JID_JOB1 do_training.slurm 