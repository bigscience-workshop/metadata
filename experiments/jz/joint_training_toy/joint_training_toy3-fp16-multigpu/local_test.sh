echo "compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 2
" > accelerate_config.yaml

accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py \
    data_config.experiment="with_metadata" \
    data_config.metadata_config.metadata_list='[entity, timestamp, url, website_description]' \
    data_config.metadata_config.max_seq_len=1024 \
    data_config.dataset_name=bs-modeling-metadata/c4-en-reduced-with-metadata \
    data_config.preprocessing_num_workers=8 \
    out_dir="metadata_outputs/${SLURM_JOB_ID}" \
    jobid="${SLURM_JOB_ID}" \
    do_train=True \
    do_eval=True \
    evaluation_strategy=STEPS \
    eval_steps=50 \
    save_strategy=STEPS \
    save_steps=51 \
