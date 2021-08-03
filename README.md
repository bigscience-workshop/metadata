# metadata
Experiments on including metadata such as URLs, timestamps, website descriptions and HTML tags during pretraining.

## Usage

```sh
accelerate launch --fp16 train.py max_train_steps=100 num_eval=1 data_config.per_device_eval_batch_size=4
```
