import logging
import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def test_toy_training_without_metadata(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            "python",
            f'{os.path.join(path_script, "metadata", "train.py")}',
            "data_config.experiment=without_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_raw_wikitext.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_raw_wikitext.jsonl")}',
            "num_eval=2",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
        ],
        stdout=subprocess.PIPE,
        stderr= subprocess.PIPE,
        universal_newlines=True,
    )
    out, out_err = process.communicate()

    # We check that the script has run smoothly
    assert process.returncode == 0, out_err

    # We could also check that the perplexity logged in wandb is below 100


def test_toy_training_with_metadata(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            "python",
            f'{os.path.join(path_script, "metadata", "train.py")}',
            "data_config.experiment=with_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_wikitext_with_metadata.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_wikitext_with_metadata.jsonl")}',
            "num_eval=2",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
        ],
        stdout=subprocess.PIPE,
        stderr= subprocess.PIPE,
        universal_newlines=True,
    )
    out, out_err= process.communicate()

    # We check that the script has run smoothly
    assert process.returncode == 0, out_err
    
    # We could also check that the perplexity logged in wandb is below 100
