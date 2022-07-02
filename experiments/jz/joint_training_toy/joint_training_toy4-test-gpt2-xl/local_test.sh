echo "compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'NO'
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: 1
" > accelerate_config.yaml

export BS=1
export SEQ=256
#accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py model_name=gpt2-xl data_config.per_device_train_batch_size=$BS data_config.per_device_eval_batch_size=$BS \
accelerate launch bsmetadata/train.py model_name=gpt2 data_config.per_device_train_batch_size=$BS data_config.per_device_eval_batch_size=$BS \
    data_config.experiment="with_metadata" \
    data_config.metadata_config.metadata_list='[entity, timestamp, url, website_description]' \
    data_config.metadata_config.max_seq_len=$SEQ \
    data_config.dataset_name=bs-modeling-metadata/c4-en-reduced-with-metadata \
    out_dir="metadata_outputs/${SLURM_JOB_ID}" \
    jobid="${SLURM_JOB_ID}" \
    do_train=True \
    do_eval=True \
    evaluation_strategy=STEPS \
    eval_steps=5900 \
    save_strategy=STEPS \
    save_steps=5900
