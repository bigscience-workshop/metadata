import logging
import os
import subprocess
import sys
from pathlib import Path
from shutil import copyfile
from unittest.mock import patch

from train import main


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def test_toy_training(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            "python",
            f'{os.path.join(path_script, "train.py")}',
            "data_config.experiment=without_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_wikitext.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_wikitext.jsonl")}',
            "num_eval=2",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
        ],
        stdout=subprocess.PIPE,
        universal_newlines=True,
    )
    out, _ = process.communicate()

    # We check that the script has run smoothly
    assert process.returncode == 0

    # We could also check that the perplexity logged in wandb is below 100
