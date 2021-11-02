

out_dir=${1:-bsmetadata/preprocessing_data} # default director: preprocessing_data

## Clone the huggingface dataset repo containing wiki dump
mkdir -p "$out_dir"
HUB_REPO_NAME=bs-modeling-metadata/wiki_dump
git clone https://huggingface.co/datasets/${HUB_REPO_NAME} $out_dir/wiki_dump


## Downloading nltk punkt to be used in sentence tokenizer
python -m nltk.downloader 'punkt'