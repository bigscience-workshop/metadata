JID_JOB1=`sbatch  load_dataset.slurm | cut -d " " -f 4`
JID_JOB2=`sbatch  --dependency=afterok:$JID_JOB1 create_dataset.slurm | cut -d " " -f 4`
sbatch  --dependency=afterok:$JID_JOB2 do_training.slurm 