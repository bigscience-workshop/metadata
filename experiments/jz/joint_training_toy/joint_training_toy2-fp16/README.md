# Experiment template

In this folder you will find script templates to run a "typical" experiment on JZ.

These scripts are designed to be run sequentially for:

1. Downloading the tokenizer and the model (`01_load_tokenizer_and_model.slurm`)
2. Downloading the dataset on a partition with internet ( `02_load_dataset.slurm`)
3. Preprocessing the dataset on a cpu-only partition (`03_create_dataset.slurm`)
4. Running the training on a gpu 16gb partition (`04_do_training.slurm`)
