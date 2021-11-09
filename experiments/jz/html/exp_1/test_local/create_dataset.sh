
BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "$BASEDIR"

# change proba
python experiments/jz/html/exp_1/start_training.py \
    --config-dir=$BASEDIR \
    data_config.train_file="/mnt/storage/Documents/hugging_face/bigscience/jz/jz-code/sync/metadata/experiments/jz/html/exp_1/test_local/inputs/sample.json" \
    data_config.validation_file=null \
    out_dir="/mnt/storage/Documents/hugging_face/bigscience/jz/jz-code/sync/metadata/experiments/jz/html/exp_1/test_local/outputs" \
    data_config.overwrite_cache=true \
    do_train=false \
    do_eval=false 