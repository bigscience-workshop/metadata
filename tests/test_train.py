import logging
import os
import subprocess
import sys
from pathlib import Path


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


def test_toy_training_without_metadata(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            sys.executable,
            f'{os.path.join(path_script, "bsmetadata", "train.py")}',
            "data_config.experiment=without_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_raw_wikitext.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_raw_wikitext.jsonl")}',
            "data_config.overwrite_cache=true",
            "eval_steps=2",
            "save_steps=2",
            "data_config.block_size=20",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out, out_err = process.communicate()

    assert "Start training" in out
    assert "Training finished" in out
    assert "checkpoint-2step.pt" in out
    assert "checkpoint-4step.pt" in out

    # We check that the script has run smoothly
    assert process.returncode == 0, f"Out: {out}\n Err:{out_err}"

    # We could also check that the perplexity logged in wandb is below 100


def test_toy_training_with_metadata(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            sys.executable,
            f'{os.path.join(path_script, "bsmetadata", "train.py")}',
            "data_config.experiment=with_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_wikitext_with_metadata.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_wikitext_with_metadata.jsonl")}',
            "data_config.overwrite_cache=true",
            "eval_steps=2",
            "save_steps=2",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out, out_err = process.communicate()

    assert "Start training" in out
    assert "Training finished" in out
    assert "checkpoint-2step.pt" in out
    assert "checkpoint-4step.pt" in out

    # We check that the script has run smoothly
    assert process.returncode == 0, f"Out: {out}\n Err:{out_err}"

    # We could also check that the perplexity logged in wandb is below 100


def test_toy_no_train_no_eval_with_metadata(tmpdir):
    os.environ["WANDB_MODE"] = "offline"

    path = Path(__file__)
    path_test_folder = path.parent.absolute()
    path_script = path.parent.parent.absolute()

    process = subprocess.Popen(
        [
            sys.executable,
            f'{os.path.join(path_script, "bsmetadata", "train.py")}',
            "data_config.experiment=with_metadata",
            f'data_config.train_file={os.path.join(path_test_folder,"data","train_toy_wikitext_with_metadata.jsonl")}',
            f'data_config.validation_file={os.path.join(path_test_folder,"data","val_toy_wikitext_with_metadata.jsonl")}',
            "data_config.overwrite_cache=true",
            "eval_steps=2",
            "save_steps=2",
            f"out_dir={tmpdir}",
            "max_train_steps=4",
            "do_train=false",
            "do_eval=false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    out, out_err = process.communicate()

    assert "Start training" not in out
    assert "Training finished" not in out

    # We check that the script has run smoothly
    assert process.returncode == 0, f"Out: {out}\n Err:{out_err}"
