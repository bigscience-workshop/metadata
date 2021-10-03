JID_JOB1=$(sbatch 01_load_tokenizer_and_model.slurm | cut -d " " -f 4)
JID_JOB2=$(sbatch --dependency=afterok:$JID_JOB1 02_load_dataset.slurm | cut -d " " -f 4)
JID_JOB3=$(sbatch --dependency=afterok:$JID_JOB2 03_create_dataset.slurm | cut -d " " -f 4)
sbatch --dependency=afterok:$JID_JOB3 04_do_training.slurm
