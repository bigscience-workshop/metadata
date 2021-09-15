python experiments/html/start_training.py \
data_config.experiment="with_metadata" \
data_config.metadata_list=["html"] \
data_config.max_seq_len=1024 \
data_config.dataset_name="SaulLu/Natural_Questions_HTML_Toy" \
data_config.train_file="nq-train-*.jsonl.gz" \
data_config.validation_file="nq-dev-*.jsonl.gz" \
data_config.extension="json" \
data_config.preprocessing_num_workers=6 \
do_train=False \
do_eval=False \
evaluation_strategy=STEPS \
eval_steps=50 \
save_strategy=STEPS \
save_steps=500 \
