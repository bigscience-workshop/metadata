import dataclasses
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import List, Optional, Union, get_args, get_origin

import hydra
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedType, DummyOptim, DummyScheduler
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.optim import AdamW
from tqdm.auto import tqdm as original_tqdm
from transformers import AddedToken, AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
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
    resume_from_checkpoint_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory where checkpoint to resume from is saved"}
    )
    model_name: str = field(default="gpt2", metadata={"help": "The name of the pretrained model to use."})
    project_name: str = field(default="metadata_lm", metadata={"help": "The project name."})
    jobid: Optional[str] = field(default=None, metadata={"help": "The jobid of the run."})
    start_with_eval: bool = field(default=False, metadata={"help": "Start by evaluating the model"})
    extra_steps_to_eval_save_at: List[int] = field(
        default_factory=(lambda: []),
        metadata={"help": "A list of additional steps to evaluate and save at."},
    )
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
    gradient_checkpointing: bool = field(
        default=False, metadata={"help": "Whether to use gradient_checkpointing to save memory."}
    )


cs = ConfigStore.instance()
cs.store(name="config", node=CFG)


def show_help(context="", cls=CFG):
    default_instance = cls()

    for field_ in dataclasses.fields(cls):
        type_ = field_.type
        origin = get_origin(type_)
        if origin is Union:  # do this to handle Optional[some_dataclass]
            type_ = get_args(type_)[0]
        if dataclasses.is_dataclass(type_):
            show_help(context=f"{context}{field_.name}.", cls=type_)
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
    # normalize first, so it doesn't overflow when there are many tokens
    normed_loss_weights = shift_mask / shift_mask.sum()
    loss = (loss * normed_loss_weights).sum()
    # per-example ppl
    # ppl = torch.exp((loss * shift_mask).sum(-1) / shift_mask.sum(-1))
    return loss


def save_model_and_tokenizer(accelerator, model, path, tokenizer=None):
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(path, save_function=accelerator.save)
    if tokenizer:
        tokenizer.save_pretrained(path, save_function=accelerator.save)


@dataclass
class TrainState:
    completed_steps: int = 0

    def step(self):
        self.completed_steps += 1

    def save(self, path):
        """to json"""
        with open(path, "w") as f:
            json.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, path):
        """from json"""
        with open(path, "r") as f:
            d = json.load(f)
        return cls(**d)


def instantiate_data_class(data_class, args):
    schema = OmegaConf.structured(data_class)
    args = OmegaConf.merge(schema, args)
    args = OmegaConf.to_object(args)
    return args


@hydra.main(config_path="hydra_configs", config_name="config")
def main(args: CFG) -> None:
    print(OmegaConf.to_yaml(args))
    # write the yaml to a file
    path = Path(args.out_dir).resolve() / "actual_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(args))
        logger.info(f"Wrote actual config to {path}")

    config_dict = OmegaConf.to_container(args)
    # The dataset library use the hash of the arguments to create the cache
    # name. Without this transformation the hash of args is not deterministic
    args = OmegaConf.to_object(args)

    # if the yaml file is used to create the config, the args are not dataclass up to this step
    # need to convert them back to dataclass via OmegaConf
    if not dataclasses.is_dataclass(args):
        args = instantiate_data_class(CFG, args)
        assert dataclasses.is_dataclass(args)
        assert dataclasses.is_dataclass(args.data_config)
        assert dataclasses.is_dataclass(args.data_config.metadata_config)

    set_seed(args.seed)
    accelerator = Accelerator()
    is_local_main_process = accelerator.is_local_main_process
    tqdm = partial(original_tqdm, disable=not is_local_main_process, position=0)
    use_deepspeed = accelerator.state.deepspeed_plugin is not None
    use_deepspeed_optimzer = use_deepspeed or "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    use_deepspeed_scheduler = use_deepspeed or "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config

    if accelerator.distributed_type == DistributedType.DEEPSPEED and not use_deepspeed_scheduler:
        assert False, "Please set scheduler in DeepSpeed config file otherwise it may not be checkpointed properly"

    os.makedirs(args.out_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name)
    config.gradient_checkpointing = args.gradient_checkpointing
    config.use_cache = not args.gradient_checkpointing  # to disable warning
    # get model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, config=config)

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
    if use_deepspeed_optimzer:
        optimizer = DummyOptim(optimizer_grouped_parameters)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-6)

    assert args.max_train_steps is not None, "max_train_steps is required, num_train_epochs is not supported"
    if use_deepspeed_scheduler:
        scheduler = DummyScheduler(
            optimizer, total_num_steps=args.max_train_steps, warmup_num_steps=args.num_warmup_steps
        )
        logger.info(
            f"Using DeepSpeed scheduler, total_num_steps={args.max_train_steps}, warmup_num_steps={args.num_warmup_steps}"
        )
    else:
        scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )
    # get dataloaders
    if args.data_config.metadata_config.local_metadata_special_token_state:
        new_tokens = list(
            chain.from_iterable(
                (start_token, end_token)
                for start_token, end_token in zip(
                    args.data_config.metadata_config.local_metadata_special_token_start.values(),
                    args.data_config.metadata_config.local_metadata_special_token_end.values(),
                )
            )
        )
        new_tokens = [
            AddedToken(token, rstrip=False, lstrip=False, single_word=False, normalized=False) for token in new_tokens
        ]
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, additional_special_tokens=new_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    train_dataloader, eval_dataloaders = get_dataloaders(tokenizer, args.data_config)

    # Prepare everything
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    eval_dataloaders = {k: accelerator.prepare(v) for k, v in eval_dataloaders.items()}
    train_state = TrainState()

    # If resume_from_checkpoint_dir is not None, we load the resumed state
    if args.resume_from_checkpoint_dir:
        path = Path(args.resume_from_checkpoint_dir).resolve()
        logger.info(f"Loading checkpoint from {path}")
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            # this is a deepspeed method, will load model, optimizer, scheduler
            # `model` wraps the optimizer and scheduler
            model.load_checkpoint(path)
        else:
            accelerator.load_state(path)
        train_state = TrainState.load(Path(path) / "train_state.json")

    # set a random dataset size if streaming
    dl_size = int(1e6) if args.data_config.streaming else len(train_dataloader)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(dl_size / args.gradient_accumulation_steps)
    # if args.max_train_steps is None:
    # args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    # Note -> the training dataloader will be shorter in multiprocess)

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

        model.train()
        if not losses:
            # in case the dataloader is empty
            return
        losses = torch.cat(losses)
        perplexity = math.exp(torch.mean(losses))
        return {"perplexity": perplexity}

    def evaluate_multiple_dateloaders(eval_dataloaders):
        for key, eval_dataloader in eval_dataloaders.items():
            logger.info(f"Evaluating split {key}")
            metrics = evaluate(eval_dataloader)
            metrics_logger.log({key: metrics})
        logger.info("Evaluation finished")

    if not args.do_train and not args.do_eval:
        return

    progress_bar = tqdm(range(args.max_train_steps), desc="training", initial=train_state.completed_steps)
    metrics_logger = Logger(is_local_main_process, project=args.project_name, config=config_dict)

    do_eval = args.do_eval and args.start_with_eval
    if do_eval:
        logger.info("Start with an evaluation")
        evaluate_multiple_dateloaders(eval_dataloaders)

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

    def save(path):
        path = Path(path).resolve()
        logger.info(f"Saving checkpoint at {path}")
        if accelerator.distributed_type == DistributedType.DEEPSPEED:
            model.save_checkpoint(path)
        else:
            accelerator.save_state(path)
        save_model_and_tokenizer(accelerator, model, path)
        if is_local_main_process:
            train_state.save(path / "train_state.json")

    step_loss = 0
    step = 0
    model.train()
    # for epoch in range(args.num_train_epochs):
    finished = False

    if not args.data_config.streaming:
        metrics_logger.log({"train_dataloader_length": len(train_dataloader)})
    while not finished:
        for batch in train_dataloader:
            step += 1
            # pop labels because we want to calculate loss ourselves
            labels = batch.pop("labels")
            metadata_mask = batch.pop("metadata_mask", None)
            outputs = model(**batch)
            batch["labels"] = labels
            loss = loss_fn(batch, outputs, metadata_mask)

            step_loss += loss.detach() / args.gradient_accumulation_steps
            if use_deepspeed:
                model.backward(loss)
                model.step()
            else:
                accelerator.backward(loss)

            do_step = step % args.gradient_accumulation_steps == 0
            if do_step:
                progress_bar.update(1)
                train_state.step()
                if not use_deepspeed:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                step_loss_gathered = accelerator.gather(step_loss).mean().item()
                metrics = {
                    "loss": step_loss_gathered,
                    "lr": max(scheduler.get_lr()),
                    "gradient_step": train_state.completed_steps,
                }
                if not args.data_config.streaming:
                    metrics["epoch"] = step / len(train_dataloader)

                metrics_logger.log(metrics)
                step_loss = 0
            else:
                continue
            completed_steps = train_state.completed_steps

            do_eval = (
                args.do_eval
                and completed_steps > 0
                and (completed_steps % eval_per_n_step == 0 or completed_steps in args.extra_steps_to_eval_save_at)
            )

            do_save = completed_steps > 0 and (
                completed_steps % save_per_n_step == 0 or completed_steps in args.extra_steps_to_eval_save_at
            )
            if do_save:
                path = Path(args.out_dir).resolve() / f"checkpoint-{completed_steps}step"
                save(path)
            if do_eval:
                evaluate_multiple_dateloaders(eval_dataloaders)

            if completed_steps >= args.max_train_steps:
                finished = True
                break
    metrics_logger.close()
    logger.info("Training finished")
    save(args.out_dir)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    newargv = []
    for arg in sys.argv:
        if arg.startswith("--local_rank"):
            pass
        else:
            newargv.append(arg)
    sys.argv = newargv

    main()
