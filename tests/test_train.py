import logging
import os
import sys
from shutil import copyfile
from unittest.mock import patch


from train import main

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def test_toy_training(tmpdir):
    os.environ["WANDB_MODE"] = "offline"
    copyfile(
        "tests/fixtures/config_local_files.yaml", os.path.join(tmpdir, "config.yaml")
    )
    path_test_folder = os.path.dirname(os.path.abspath(__file__))
    testargs = [
        "",
        f"train_file={path_test_folder}/data/train_toy_wikitext.jsonl",
        f"validation_file={path_test_folder}/data/val_toy_wikitext.jsonl",
        "num_eval=2",
        f"out_dir={tmpdir}",
    ]
    with patch.object(sys, "argv", testargs):
        main()
