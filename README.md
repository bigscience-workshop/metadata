# Metadata

Experiments on including metadata such as URLs, timestamps, website descriptions and HTML tags during pretraining.

## Usage

```sh
accelerate launch --fp16 train.py max_train_steps=100 num_eval=1 data_config.per_device_eval_batch_size=4
```

## Get Help

```sh
python metadata/train.py [-h] [--help]
```


## Metadata format

```javascript
{
    "id": "ABC",
    "text": "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is Yafai has got to go him.",
    "metadata": [
        {
            "key": "url",
            "type": "global",
            "value": "https://www.bbc.com/sport/live/olympics/50974152"
        },
        {
            "key": "timestamp",
            "type": "global",
            "value": "2018-12-10T13:45:00.000Z"
        },
        {
            "key": "entity",
            "type": "local",
            "char_start_idx": 132,
            "char_end_idx": 137,
            "value": "Galal Yafai"
        }
    ]
}
```

## Contribute 🧠

After installing the development dependencies, first you need to install the package in editable mode. You can do it by running in a bash at the root of the repository

```
pip install -e .
```

and then you can execute the tests by running:

```sh
python -m pytest .
```

In order to have a unified code style, we have implemented some formatting tools. Before you commit or PR, it would be great if you could run:

```sh
make style && make quality
```
