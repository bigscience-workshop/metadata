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
    weight_decay: float = field(default=0.0, metadata={"help": "The weight decay to use for training."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of gradient accumulation steps to perform before updating model parameters."},
    )
    num_train_epochs: int = field(default=1, metadata={"help": "The number of epochs to train the model for."})
    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of training steps (overrides num_train_epochs)."}
    )
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The type of learning rate schedule to use."})
    num_warmup_steps: int = field(
        default=1000, metadata={"help": "The number of warmup steps during which the learning rate is increased."}
    )
    seed: int = field(default=42, metadata={"help": "The seed used for RNG initialization."})
    out_dir: str = field(
        default="output_dir", metadata={"help": "The output directory in which the trained model is saved."}
    )
    model_name: str = field(default="gpt2", metadata={"help": "The name of the pretrained model to use."})
    project_name: str = field(default="metadata_lm", metadata={"help": "The project name."})
    jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the run."})
    start_with_eval: bool = field(default=False, metadata={"help": "Start by evaluating the model"})
    evaluation_strategy: IntervalStrategy = field(
        default="STEPS",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_num_per_epoch: int = field(
        default=3,
        metadata={
            "help": "If evaluation strategy is `epoch`. The number of evaluations to perform per epoch during training."
        },
    )
    eval_steps: int = field(
        default=100, metadata={"help": "If evaluation strategy is `steps`. Run an evaluation every X steps."}
    )

    save_strategy: IntervalStrategy = field(
        default="STEPS",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_num_per_epoch: int = field(
        default=3,
        metadata={"help": "If save strategy is `epoch`. The number of savings to perform per epoch during training."},
    )
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X update steps."})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
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
    accelerator = Accelerator()
    args.data_config.distributed_type = accelerator.distributed_type

    print(OmegaConf.to_yaml(args))
    config_dict = OmegaConf.to_container(args)

    # The dataset library use the hash of the arguments to create the cache
    # name. Without this transformation the hash of args is not deterministic
    args = OmegaConf.to_object(args)

    set_seed(args.seed)
    is_local_main_process = accelerator.is_local_main_process
    tqdm = partial(original_tqdm, disable=not is_local_main_process, position=0)

    os.makedirs(args.out_dir, exist_ok=True)

    # get dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataloader, eval_dataloaders = get_dataloaders(tokenizer, args.data_config)

    # get model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    # Note -> the training dataloader needs to be prepared before we grab its length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.evaluation_strategy == IntervalStrategy.EPOCH and args.eval_num_per_epoch >= 1:
        eval_per_n_step = args.max_train_steps // (args.eval_num_per_epoch * args.num_train_epochs)
    elif args.evaluation_strategy == IntervalStrategy.STEPS:
        eval_per_n_step = args.eval_steps
    else:  # IntervalStrategy.NO or (args.eval_num_per_epoch < 1 and args.evaluation_strategy == IntervalStrategy.EPOCH)
        eval_per_n_step = args.max_train_steps + 1  # will never eval

    if args.save_strategy == IntervalStrategy.EPOCH and args.save_num_per_epoch >= 1:
        save_per_n_step = args.max_train_steps // (args.save_num_per_epoch * args.num_train_epochs)
    elif args.save_strategy == IntervalStrategy.STEPS:
        save_per_n_step = args.save_steps
    else:  # IntervalStrategy.NO or (args.save_num_per_epoch < 1 and args.save_strategy == IntervalStrategy.EPOCH)
        save_per_n_step = args.max_train_steps + 1  # will never eval

    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

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

    if not args.do_train and not args.do_eval:
        return

    progress_bar = tqdm(range(args.max_train_steps), desc="training")
    completed_steps = 0
    metrics_logger = Logger(is_local_main_process, project=args.project_name, config=config_dict)

    do_eval = args.do_eval and args.start_with_eval
    if do_eval:
        logger.info("Start with an evaluation")
        for key, eval_dataloader in eval_dataloaders.items():
            metrics = evaluate(eval_dataloader)
            metrics_logger.log({key: metrics})
        logger.info("Evaluation finished")

    if not args.do_train:
        return

    logger.info("Start training")
    logger.info(
        f"  Evaluation will be done every {eval_per_n_step} steps, "
        f"for a total of {args.max_train_steps//eval_per_n_step} times."
    )
    logger.info(
        f"  Saving will be done every {save_per_n_step} steps, "
        f"for a total of {args.max_train_steps//save_per_n_step} times."
    )
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # pop labels because we want to calculate loss ourselves
            labels = batch.pop("labels")
            metadata_mask = batch.pop("metadata_mask", None)
            outputs = model(**batch)
            batch["labels"] = labels
            loss = loss_fn(batch, outputs, metadata_mask)

            metrics_logger.log({"loss": loss})
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            do_step = step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1
            if do_step:
                #             accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                metrics_logger.log({"gradient_step": completed_steps})
            else:
                continue

            do_eval = args.do_eval and completed_steps > 0 and completed_steps % eval_per_n_step == 0
            if do_eval:
                for key, eval_dataloader in eval_dataloaders.items():
                    metrics = evaluate(eval_dataloader)
                    metrics_logger.log({key: metrics})
                    # logger.info(f"epoch {epoch}: perplexity: {perplexity}")

            do_save = is_local_main_process and completed_steps > 0 and completed_steps % save_per_n_step == 0
            if do_save:
                save_path = os.path.join(args.out_dir, f"checkpoint-{completed_steps}step.pt")
                logger.info(f"Save model at {save_path}")
                save_dict = {
                    "epoch": epoch + 1,
                    "state_dict": accelerator.unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                torch.save(save_dict, save_path)
                del save_dict
                gc.collect()

            if completed_steps >= args.max_train_steps:
                break
    metrics_logger.close()
    logger.info("Training finished")

    if is_local_main_process and args.out_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.out_dir, save_function=accelerator.save)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
