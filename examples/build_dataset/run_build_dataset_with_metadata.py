import dataclasses
import json
import sys
import typing

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from bsmetadata.experiments.with_metadata import build_dataset
from bsmetadata.input_pipeline import DataConfig


cs = ConfigStore.instance()
cs.store(name="config", node=DataConfig)


def show_help(cls, context=""):
    default_instance = cls()
    for field_ in dataclasses.fields(cls):
        if dataclasses.is_dataclass(field_.type):
            show_help(context=f"{context}{field_.name}.", cls=field_.type)
        elif (
            hasattr(field_.type, "__origin__")
            and field_.type.__origin__ == typing.Union
            and hasattr(field_.type, "__args__")
            and len(field_.type.__args__) > 0
        ):
            for arg in field_.type.__args__:
                if dataclasses.is_dataclass(arg):
                    show_help(context=f"{context}{field_.name}.", cls=arg)

        else:
            kwargs = field_.metadata.copy()
            help = kwargs.get("help", "")
            default = getattr(default_instance, field_.name)  # init and tell the default
            print(f"{context}{field_.name}: {help} (default={json.dumps(default)})")


@hydra.main(config_path=None, config_name="config")
def main(args: DataConfig) -> None:
    print(OmegaConf.to_yaml(args))

    args = OmegaConf.to_object(args)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    build_dataset(tokenizer, args)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help(cls=DataConfig)
        sys.exit()
    main()
