# README for training

This is a document for all the training related information.

## Huggingface org

You need to join https://huggingface.co/bs-modeling-metadata to be able to access the training data.

## Dataset

The whole dataset is in https://huggingface.co/bs-modeling-metadata/c4-en-html-with-metadata

- Some files of the dataset have entity metadata, the rest does not. You can find the list of files with entity metadata in the code datasetv2.py

- Currently we use tensorflow to preprocess the dataset. The code requires the dataset to be accessible locally (downloaded or mounted). For my local testing, I only use a subset of the dataset.
    - The code is in with_metadata_datasetv2_tf.py

### Test set
For historical reasons, the test set is one of the file in the dataset.
It is the file `c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz` in the dataset. ***not sure about this*** need to ask @Mike.
And we make a copy of it for each metadata type and removes the examples that does not have the metadata type.
The datasets for each metadata type have this naming: https://huggingface.co/bs-modeling-metadata/c4-en-html-with-validation_metadata_website_desc


### Training set

- The training set is the whole dataset minus the test set. You need to carefully specify it during training to exclude it.

## Training

### Code

https://github.com/bigscience-workshop/metadata

### Environment

I suggest to use conda to create an environment with a python>=3.8
Install pytorch with the appropriate cuda version. For me I used
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
Then install the requirements with `pip install -r requirements.txt`
And install the package with `pip install -e '.[preprocessing]'`
(Some of the requirements are necessary for training, but may be imported in the code. I'm too lazy to check which ones are necessary and make the import optional)

And install certain version of accelerate and deepspeed with
`pip install accelerate==0.10.0 deepspeed==0.7.7`
(There are newer versions, but they are not compatible with the current code. I'm not sure if we should update the code or not)

`pip install xxhash`

### Configurations

- We use accelerate deepspeed for training. The deepspeed config that we are passed to accelerate is in `bsmetadata/deepspeed_configs/v2.json` and the other configs are handled using hydra and the config file is in `bsmetadata/hydra_configs/v2.yaml`

- some of the configs are duplicated.
- Important: (this can fail silently!) You need to make sure gradient_accumulation_steps are the same in both configs.
- batch size are also in both configs.
 - per_device_eval_batch_size, per_device_train_batch_size:  in the hydra config
 - train_micro_batch_size_per_gpu: in the deepspeed config
- this list may not be exhaustive

- you can overwrite hydra configs in the command line. But I think it is better to update the config file directly for serious training.

- The example launch script is `experiments/hpsearch/test.sh`

- The entrypoint python file is `bsmetadata/train.py`

- The adjust number of gpus in the launch script. And make corresponding changes in the deepspeed config (and the hydra config if you change the gradient_accumulation_steps)

- Currently this branch contains config that can be run on 2 RTX 3090 GPUs. You can use it as a starting point to run on other GPUs. Using only html metadata.

#### Important configs

- metadata_list and metadata_column_list
    -  double check with @Timo for what we want to use

- local_metadata_special_tokens
    -  do we want to add special token? what's the implication of this on evaluation?
- metadata_probability
    -  I think the effect of this is not a lot. Probably we can just keep it at 0.5
- random_sample_metadata_weights
    - Do we want a different sampling logic? Maybe we can instead sample each kind of metadata independently?
- validation_file
    - Make sure to specify the correct validation file, the same one as the test set.

- Learning rate, warmup, etc
    - I think the current code uses deepspeed's optimizer and scheduler. So you need to change the deepspeed config to change the learning rate, warmup, etc. I'm not sure if the hydra config is used at all.

#### Performance tuning (optimize training efficiency)

- gradient_checkpointing: you can toggle this in the hydra config.
- cpu_offload: You can disable this in the deepspeed config. But I need it for my 3090 GPUs.
- zero_optimization: You can try different values for this in the deepspeed config. I don't fully understand this.

#### Resume training (preemptive training)

- The current checkpoint logic doesn't have delete logic. It simply save a checkpoint every n steps. In addition to that, you can spedify `extra_steps_to_eval_save_at` in hydra config.
- You can continue training from a checkpoint with resume_from_checkpoint_dir
    - It'll re-iterate over the dataset. This may take some time.
    - There is this new library https://github.com/mosaicml/streaming, which can be used to stream the dataset and avoid re-iterating over the dataset. But I haven't tried it yet. It requires to convert the dataset to a different format.

- Currently we don't have evaluation during training. But we want to add it. We had it in the previous version of the code, it's currently just evaluating an empty dataloader. So it should be easy to add it back.
- We have a script to evaluate the model on the test set. It is in `bsmetadata/evaluate.py` and you'd need to add the code to call it in the training loop (`bsmetadata/train.py`)
- The evaluation script is not very efficient. It doesn't use batch or multi GPU, but it only takes ~10 minutes to evaluate it.
- TODO: Need to make sure the evaluation script is up to date. @Nikolas @Mike