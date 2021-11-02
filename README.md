# BigScience Modeling Metadata

This repository contains code for including metadata such as URLs, timestamps, website descriptions and HTML tags during language model pretraining. The purpose is to explore, solely from a modeling perspective, how to make good use of metadata to improve various aspects of the model (such as its zero-shot text generation abilities). This repository is **not** intended for general contributions to metadata that are not concerned with modeling.

## Usage

```sh
accelerate launch --fp16 train.py max_train_steps=100 eval_num_per_epoch=1 data_config.per_device_eval_batch_size=4
```

## Get Help

```sh
python bsmetadata/train.py [-h] [--help]
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

And there are also two optional keys (which must be set or not set for all metadata sharing the same ``key`` value):
- ``relative_start_pos`` position at which the start metadata is placed at a given character index. 
- ``relative_end_pos`` position at which the end metadata is placed at a given character index.
The counter is common between ``relative_start_pos`` and ``relative_end_pos`` for a given ``key`` value.
### Example

Assume that the following text is extracted from `https://www.bbc.com/sport/live/olympics/50974152`, which is an article that was published on `2018-12-10T13:45:00.000Z` (line 2-3 show the output of an entity tagger applied to this text).

```html
<body><div><p>It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is <a>Yafai</a> has got to go him.</p>\n</div></body>
                                                                                                                                                     ^^^^^
                                                                                                                                                     Entity: Galal Yafai
```

This text would be represented as the following input example with two global metadata instances (``url`` and ``timestamp``) and five local metadata instances (1 ``entity`` and 4 ``html``). Note that this entire input should be in *a single line* in the actual dataset.

```javascript
{
    "text": "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is Yafai has got to go him.\n",
    "metadata": [
        {"key": "url", "type": "global", "value": "https://www.bbc.com/sport/live/olympics/50974152"},
        {"key": "timestamp", "type": "global", "value": "2018-12-10T13:45:00.000Z"},
        {"key": "entity", "type": "local", "char_start_idx": 132, "char_end_idx": 137, "value": "Galal Yafai"},
        {'key': 'html', 'type': 'local', 'char_start_idx': 132, 'relative_start_pos': 0, 'char_end_idx': 137, 'relative_end_pos': 0, 'value': 'a', 'html_attrs': {'attrs': [], 'values': []}},
        {'key': 'html', 'type': 'local', 'char_start_idx': 0, 'relative_start_pos': 2, 'char_end_idx': 156, 'relative_end_pos': 0, 'value': 'p', 'html_attrs': {'attrs': [], 'values': []}},
        {'key': 'html', 'type': 'local', 'char_start_idx': 0, 'relative_start_pos': 1, 'char_end_idx': 157, 'relative_end_pos': 0, 'value': 'div', 'html_attrs': {'attrs': [], 'values': []}},
        {'key': 'html', 'type': 'local', 'char_start_idx': 0, 'relative_start_pos': 0, 'char_end_idx': 157, 'relative_end_pos': 1, 'value': 'body', 'html_attrs': {'attrs': [], 'values': []}},
    ]
}
```

Below is a table showing ``relative_start_pos`` (`start`) and ``relative_end_pos`` (`end`) fields work on some HTML tags examples.


| Input | `start` (`<i>`) | `end`(`<i>`) | `start` (`<b>`) | `end`(`<b>`) |
| - | - | - | - | - |
| ``<i></i><b> ... </b>`` | 0 | 1 | 2 | 0 |
| ``<i><b> ... </b></i>`` | 0 | 1 | 1 | 0 |
| ``<i> ... </i><b></b>`` | 0 | 0 | 1 | 2 |
| ``<i> ... <b></b></i>`` | 0 | 2 | 0 | 1 |

## Pre-processing Metadata:

### Entity Tags.

**Pre-requisite steps for preprocessing Entity Tags**

* Run `pip install -r bsmetadata/vendor/REL/requirements.txt`
* Run `preprocessing_scripts/download_entity_processing_files.sh` to download all the required files.

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
