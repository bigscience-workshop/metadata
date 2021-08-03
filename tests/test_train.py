import sys
import os
import logging
import inspect
from shutil import copyfile
import torch
from transformers.testing_utils import torch_device
from hydra.experimental import initialize
import hydra
from unittest.mock import patch

from train import main


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def test_toy_training(tmpdir):
    os.environ["WANDB_MODE"] = "offline"
    copyfile("tests/fixtures/config_local_files.yaml", os.path.join(tmpdir, "config.yaml"))
    print(tmpdir)
    # os.chdir(tmpdir)
    path_train_fn = os.path.abspath(inspect.getfile(main))
    relative_path = os.path.relpath(tmpdir, path_train_fn)
    # hydra._internal.hydra.GlobalHydra.get_state().clear()
    # with initialize(config_path=relative_path):
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
