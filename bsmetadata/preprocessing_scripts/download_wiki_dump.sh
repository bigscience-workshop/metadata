

HUB_REPO_NAME=bs-modeling-metadata/wiki_dump
git clone https://huggingface.co/datasets/${HUB_REPO_NAME}


# Downloading nltk punkt to be used in sentence tokenizer
python -m nltk.downloader 'punkt'