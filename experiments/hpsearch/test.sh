export MODEL=gpt2-xl
export NUM_GPU=8

export DEEPSPEED_CONFIG=$(realpath bsmetadata/deepspeed_configs/v2.json)
echo "deepspeed_config_file: $DEEPSPEED_CONFIG"
echo "compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: $DEEPSPEED_CONFIG
distributed_type: DEEPSPEED
fp16: true
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
num_machines: 1
num_processes: $NUM_GPU 
mixed_precision: fp16
" > accelerate_config.yaml

accelerate launch --config_file accelerate_config.yaml bsmetadata/train.py --config-name v2 \
  model_name=$MODEL \
    data_config.train_file='c4-en-html_cc-main-2019-18_pq0*-010.jsonl.gz' \
    data_config.validation_file='c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz' \
    data_config.preprocessing_num_workers=48  extra_steps_to_eval_save_at='[2]' \
    data_config.streaming=True out_dir=/mnt/ssd-1/bigscience-metadata/stream10file
    #out_dir=/mnt/ssd-1/bigscience-metadata/run1
    #data_config.train_file='c4-en-html_cc*.jsonl.gz' data_config.streaming=True out_dir=/mnt/ssd-1/bigscience-metadata/run1
