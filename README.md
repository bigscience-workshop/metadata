# BigScience Metadata

This repository contains code for including metadata such as URLs, timestamps, website descriptions and HTML tags during language model pretraining.

## Usage

```sh
accelerate launch --fp16 train.py max_train_steps=100 num_eval=1 data_config.per_device_eval_batch_size=4
```

## Get Help

```sh
python metadata/train.py [-h] [--help]
```

## Metadata Format

This script expects metadata to be in [JSON lines (.jsonl)](https://jsonlines.org/) format. Each JSON line is required to have the following fields:

- ``text``: The actual input text.
- ``metadata``: A list of metadata associated with the given text.

The script supports two different kinds of metadata: *global* metadata, which applies to the whole text, and *local* metadata, which applies only to parts of it.

### Global Metadata

Global metadata is required to have the following fields:

- ``key``: A unique key to identify this kind of metadata (e.g., ``url`` or ``timestamp``).
- ``type``: This must be set to ``global``.
- ``value``: The actual value associated with this metadata instance (e.g., an actual URL or timestamp).

### Local Metadata

Local metadata is required to have the following fields:

- ``key``: A unique key to identify this kind of metadata (e.g., ``entity`` or ``html``).
- ``type``: This must be set to ``local``.
- ``char_start_idx``: The index of the first character in ``text`` that is associated with this metadata instance.
- ``char_end_idx``: The index of the first character in ``text`` that is **not** associated with this metadata instance.
- ``value``: The actual value associated with this metadata instance (e.g., an entity name or HTML tag).

### Example

Below is a valid input example consisting of a text with two global metadata instances (``url`` and ``timestamp``) and one local metadata instance (``entity``).
Note that this entire input should be in *a single line* in the actual dataset.

```javascript
{
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
