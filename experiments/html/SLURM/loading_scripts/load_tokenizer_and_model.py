import hydra
import sys
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.train import show_help, CFG

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