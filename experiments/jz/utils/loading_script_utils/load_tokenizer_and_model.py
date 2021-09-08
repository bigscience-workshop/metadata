import logging
import sys

import hydra
import transformers.utils.logging as logging_transformers
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.train import CFG, show_help


# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO
)
logging_transformers.set_verbosity_info()
logging_transformers.enable_default_handler()
logging_transformers.enable_explicit_format()

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config", node=CFG)


@hydra.main(config_path=None, config_name="config")
def main(args: CFG) -> None:
    # get dataloaders
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # get model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
