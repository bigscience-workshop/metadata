import dataclasses
import gc
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import hydra
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from datasets.features import Value
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.auto import tqdm as original_tqdm
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from transformers.trainer_utils import IntervalStrategy

from bsmetadata.input_pipeline import DataConfig, get_dataloaders


logger = logging.getLogger(__name__)


@dataclass
class CFG:
    data_config: DataConfig = DataConfig()
    out_dir: str = field(
        default="output_dir", metadata={"help": "The output directory in which the trained model is saved."}
    )
    training_jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the training run."})
    jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the evaluation."})
    checkpoints_to_evaluate: str = field(
        default="all",
        metadata={
            "help": "Indicate whether all checkpoints should be evaluated ('all') or only the last one ('last')"
        },
    )
    eval_name: str = field(
        default="ppl on val without metadata",
        metadata={
            "help": "Indicate whether all checkpoints should be evaluated ('all') or only the last one ('last')"
        },)
    seed: int = field(default=42, metadata={"help": "The seed used for RNG initialization."})
    model_name: str = field(default="gpt2", metadata={"help": "The name of the pretrained model to use."})
    project_name: str = field(default="metadata_lm", metadata={"help": "The project name."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})


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

    set_seed(args.seed)
    accelerator = Accelerator()
    is_local_main_process = accelerator.is_local_main_process
    tqdm = partial(original_tqdm, disable=not is_local_main_process, position=0)

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

    # todo trick 
    _ , eval_dataloaders = get_dataloaders(tokenizer, args.data_config)
    logger.info("The dataloaders have been build")

    if not args.do_eval:
        return

    # get model
    logger.info("Load model")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Prepare everything
    model = accelerator.prepare(model)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    # Note -> the training dataloader needs to be prepared before we grab its length below (cause its length will be
    # shorter in multiprocess)

    @torch.no_grad()
    def evaluate(eval_dataloader):
        model.eval()
        losses = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="eval")):  # , leave=False)
            labels = batch.pop("labels")
            metadata_mask = batch.pop("metadata_mask", None)
            outputs = model(**batch)
            batch["labels"] = labels
            loss = loss_fn(batch, outputs, metadata_mask)

            losses.append(accelerator.gather(loss.repeat(args.data_config.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        perplexity = math.exp(torch.mean(losses))
        model.train()
        return {"perplexity": perplexity}

    logger_metrics = Logger(is_local_main_process, project=args.project_name, config=args)

    checkpoint_names = sorted(
        [
            os.path.join(args.out_dir, args.training_jobid, file_name)
            for file_name in os.listdir(os.path.join(args.out_dir, args.training_jobid))
            if file_name.split(".")[-1] == "pt" and file_name.split("-")[0] == "checkpoint"
        ]
    )
    if args.checkpoints_to_evaluate == "last":
        checkpoint_names = [checkpoint_names[-1]]
    elif args.checkpoints_to_evaluate != "all":
        raise ValueError("Wrong argument set for 'checkpoints_to_evaluate', valid possibilities are 'all' or 'last'.")

    logger.info(f"Will evaluate the following checkpoints: {checkpoint_names}")
    for file_name in checkpoint_names:
        checkpoint_path = os.path.join(args.out_dir, args.jobid, file_name)
        step = file_name.split(".")[0].split("-")[-1].split("step")[0]
        logger.info(f"Loading state dict for the checkpoint of step {step}")
        state_dict = torch.load(checkpoint_path)["state_dict"]
        logger.info("Loading state dict finished")

        model.load_state_dict(state_dict)

        logger.info(f"***** Evaluation step {step} *****")
        for key, eval_dataloader in eval_dataloaders.items():
            metrics = evaluate(eval_dataloader)
            logger_metrics.log({f"{args.eval_name} {key}": metrics, "step": step})
        # logger_metrics.info(f"epoch {epoch}: perplexity: {perplexity}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
