import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from git import Repo
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb


logger = logging.getLogger(__name__)


@dataclass
class CFG:
    out_dir: str = field(
        default="output_dir", metadata={"help": "The output directory in which the trained model is saved."}
    )
    training_jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the training run."})
    jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the evaluation."})
    checkpoints_to_commit: str = field(
        default="all",
        metadata={
            "help": "Indicate whether all checkpoints should be evaluated ('all') or only the last one ('last')"
        },
    )
    model_name: str = field(default="gpt2", metadata={"help": "The name of the pretrained model to use."})
    local_repo_path: Optional[str] = field(default=None, metadata={"help": "The path to the repository"})


cs = ConfigStore.instance()
cs.store(name="config", node=CFG)


def show_help(context="", cls=CFG):
    default_instance = cls()
    for field_ in dataclasses.fields(cls):
        if dataclasses.is_dataclass(field_.type):
            show_help(context=f"{context}{field_.name}.", cls=field_.type)
        else:
            kwargs = field_.metadata.copy()
            # print(field)
            help = kwargs.get("help", "")
            default = getattr(default_instance, field_.name)  # init and tell the default
            print(f"{context}{field_.name}: {help} (default={json.dumps(default)})")


class Logger:
    def __init__(self, is_local_main_process, *args, **kwargs):
        self.is_local_main_process = is_local_main_process
        if self.is_local_main_process:
            self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        if self.is_local_main_process:
            wandb.log(dic)

    def close(self):
        if self.is_local_main_process:
            wandb.finish()


def loss_fn(batch, outputs, metadata_mask=None):
    b = outputs.logits.size(0)
    lm_logits = outputs.logits
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if metadata_mask is not None:
        loss_mask = torch.logical_and(attention_mask, ~metadata_mask)
    else:
        loss_mask = attention_mask
    shift_mask = loss_mask[..., 1:].contiguous()
    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(b, -1)
    loss = (loss * shift_mask).sum() / shift_mask.sum()
    # per-example ppl
    # ppl = torch.exp((loss * shift_mask).sum(-1) / shift_mask.sum(-1))
    return loss


@hydra.main(config_path=None, config_name="config")
def main(args: CFG) -> None:
    print(OmegaConf.to_yaml(args))

    # The dataset library use the hash of the arguments to create the cache
    # name. Without this transformation the hash of args is not deterministic
    args = OmegaConf.to_object(args)

    accelerator = Accelerator()
    is_local_main_process = accelerator.is_local_main_process

    os.makedirs(args.out_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_local_main_process else logging.WARN,
    )

    # get dataloaders
    logger.info("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("Load dataloaders")

    # get model
    logger.info("Load model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Prepare everything
    model = accelerator.prepare(model)

    checkpoint_names_dict = {}
    path = os.path.join(args.out_dir, args.training_jobid)
    for file_name in os.listdir(path):
        if file_name.split(".")[-1] == "pt" and file_name.split("-")[0] == "checkpoint":
            step = file_name.split(".")[0].split("-")[-1].split("step")[0]
            checkpoint_names_dict[int(step)] = os.path.join(path, file_name)

    steps = sorted(list(checkpoint_names_dict.keys()))
    if args.checkpoints_to_commit == "last":
        steps = [steps[-1]]
    elif args.checkpoints_to_commit != "all":
        raise ValueError("Wrong argument set for 'checkpoints_to_evaluate', valid possibilities are 'all' or 'last'.")

    logger.info(f"Will add the following checkpoints: {checkpoint_names_dict}")

    local_repo_path = args.local_repo_path
    repo = Repo(local_repo_path)
    repo.config_writer().set_value("user", "name", "SaulLu").release()
    repo.config_writer().set_value("user", "email", "lucilesaul.com@gmail.com").release()

    tokenizer.save_pretrained(local_repo_path)
    for step in steps:
        checkpoint_path = checkpoint_names_dict[step]
        logger.info(f"Loading state dict for the checkpoint of step {step}")
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        logger.info("Loading state dict finished")

        model.load_state_dict(state_dict)
        model.save_pretrained(local_repo_path)
        # Add and commit to new branch
        new_branch = f"checkpoint-step-{step}"
        current = repo.create_head(new_branch)
        current.checkout()

        if repo.index.diff(None) or repo.untracked_files:
            repo.git.add(A=True)
            repo.git.commit(m=f"add model at step {step}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
