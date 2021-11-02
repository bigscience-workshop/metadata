

${1:-../preprocessing_data} # default director: preprocessing_data

## Clone the huggingface dataset repo containing wiki dump
mkdir $1
HUB_REPO_NAME=bs-modeling-metadata/wiki_dump
git clone https://huggingface.co/datasets/${HUB_REPO_NAME} $1


## Downloading nltk punkt to be used in sentence tokenizer
python -m nltk.downloader 'punkt'