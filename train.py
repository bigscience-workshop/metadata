import gc
import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Optional

import datasets
import hydra
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from tqdm.auto import tqdm as original_tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoTokenizer,
                          get_scheduler, set_seed)

from input_pipeline import DataConfig, get_dataloaders


@dataclass
class TrainingArguments:
    per_device_eval_batch_size: int = 2
    per_device_train_batch_size: int = 2
    weight_decay: float = 0.0
    learning_rate: float = 5e-5
    gradient_accumulation_steps: int = 1
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None
    lr_scheduler_type: str = "linear"
    num_warmup_steps: int = 1000
    seed: int = 42
    out_dir: str = "output_dir"
    num_eval: int = 3


@dataclass
class CFG(DataConfig, TrainingArguments):
    # data_config: DataConfig = DataConfig()
    # training_args: TrainingArguments= TrainingArguments()
    h: bool = False  # help, print config and exit
    model_name: str = "gpt2"
    project_name: str = "metadata_lm"


cs = ConfigStore.instance()
cs.store(name="config", node=CFG)


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


@hydra.main(config_path="conf", config_name="config")
def main(args: CFG) -> None:
    print(OmegaConf.to_yaml(args))
    if args.h:
        sys.exit()

    set_seed(args.seed)
    accelerator = Accelerator()
    is_local_main_process = accelerator.is_local_main_process
    tqdm = partial(original_tqdm, disable=not is_local_main_process)

    # post-process args
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    os.makedirs(args.out_dir, exist_ok=True)

    # get dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataloader, eval_dataloaders = get_dataloaders(tokenizer, args)

    # get model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}

    # Note -> the training dataloader needs to be prepared before we grab its length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    eval_per_n_step = args.max_train_steps // args.num_eval
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
        for step, batch in enumerate(
            tqdm(eval_dataloader, desc="eval")
        ):  # , leave=False)
            labels = batch.pop("labels")
            metadata_mask = batch.get("metadata_mask", None)
            outputs = model(**batch)
            batch["labels"] = labels
            loss = loss_fn(batch, outputs, metadata_mask)

            losses.append(
                accelerator.gather(loss.repeat(args.per_device_eval_batch_size))
            )

        losses = torch.cat(losses)
        perplexity = math.exp(torch.mean(losses))
        model.train()
        return {"perplexity": perplexity}

    progress_bar = tqdm(range(args.max_train_steps), desc="training")
    completed_steps = 0
    logger = Logger(
        is_local_main_process, project=args.project_name, config=args, dir=args.out_dir
    )
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            # pop labels because we want to calculate loss ourselves
            labels = batch.pop("labels")
            metadata_mask = batch.get("metadata_mask", None)
            outputs = model(**batch)
            batch["labels"] = labels
            loss = loss_fn(batch, outputs, metadata_mask)

            logger.log({"loss": loss})
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)

            do_step = (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            )
            do_eval = completed_steps > 0 and completed_steps % eval_per_n_step == 0
            if do_step:
                #             accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
            else:
                continue
            if do_eval:
                for key, eval_dataloader in eval_dataloaders.items():
                    metrics = evaluate(eval_dataloader)
                    logger.log({key: metrics})

                # logger.info(f"epoch {epoch}: perplexity: {perplexity}")
                if is_local_main_process:
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    torch.save(
                        save_dict,
                        os.path.join(
                            args.out_dir, f"checkpoint-{completed_steps}step.pt"
                        ),
                    )
                    del save_dict
                    gc.collect()
            if completed_steps >= args.max_train_steps:
                break
    logger.close()

    if is_local_main_process and args.out_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.out_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
