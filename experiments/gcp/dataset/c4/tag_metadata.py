import json
import os
import shutil
import sys
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial, reduce
from pprint import pformat
from typing import List, Optional, Tuple, Union

import hydra
from datasets import Dataset, Features, Value, config, load_dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from loguru import logger
from omegaconf import OmegaConf

import wandb
from bsmetadata.preprocessing_utils import (
    DatasourcePreprocessor,
    EntityParagraphPreprocessor,
    EntityPreprocessor,
    ErrorWrapperPreprocessor,
    GenerationLengthPreprocessor,
    HtmlPreprocessor,
    MetadataTagger,
    ParagraphPreprocessor,
    TimestampPreprocessor,
    TitlePreprocessor,
    UrlPreprocessor,
    WebsiteDescPostprocessor,
    WebsiteDescPreprocessor,
)


@dataclass
class PreprocessingConfig:
    task_id: int = field(metadata={"help": "The id of the task"})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    num_files_to_process: int = field(default=1, metadata={"help": "the number of files to process"})
    path_wiki_db: Optional[str] = field(
        default=None,
        metadata={"help": "The path to the wikipedia database file necessary for the website descriptions"},
    )
    entity_path_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to the directory containing the directories `ed-wiki-2019`, `generic` and `wiki_2019` "
        },
    )
    path_or_url_flair_ner_model: Optional[str] = field(
        default=None, metadata={"help": "The path or name of the flair ner model to use to preprocess entities"}
    )
    metadata_to_include: Optional[list] = field(
        default_factory=list,
        metadata={"help": "The list of metadata to tag"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    dataset_revision: Optional[str] = field(
        default="main", metadata={"help": "The revision of the dataset to use (via the datasets library)"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether the local cache containing datasets should be overwritten."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3?"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    map_batch_size: Optional[int] = field(
        default=64,
        metadata={
            "help": "This is the size of the batch size that will be used for the mapping operation when generating"
            " the dataset. If you are using `with_metadata` the recommended batch size is 1.."
        },
    )
    project_name: str = field(default="tag_metadata", metadata={"help": "The project name."})
    select_n_first_indices: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of indices to process from the initial datasets. If None, the "
            "whole dataset will be processed. Use for debugging purpose."
        },
    )
    use_load_from_disk: bool = field(
        default=False,
        metadata={
            "help": "If false, the program will load the dataset with `load_dataset` and if true, it will load it "
            "with `load_from_disk`."
        },
    )
    skip_if_save_file_already_exist: bool = field(
        default=False,
        metadata={
            "help": "If true, the program will process the file if the path at which the final dataset will be saved already exist."
        },
    )
    set_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the dataset"},
    )
    download_mode: Optional[str] = field(
        default=None,  # It was "force_redownload"
        metadata={"help": "load_dataset()'s download_mode"},
    )
    save_as_jsonl_gz: Optional[bool] = field(
        default=True,
        metadata={"help": "If true, the program will save the processed dataset as a `jsonl.gz` file."},
    )
    save_as_jsonl_gz_batch_size: int = field(
        default=64,
        metadata={"help": "The number of records per batch to save a byte-stream of a `jsonl.gz` file."},
    )
    save_as_jsonl_gz_num_workers: Optional[int] = field(
        default=8, metadata={"help": "The number of processes to use for savng a `jsonl.gz` file."}
    )
    file_pq_serial_range: Optional[Tuple[int, int, int]] = field(
        default=None,
        metadata={
            "help": (
                "Process files of the same pq?? with serials in a closed interval [???, ???].  "
                "For example, `file_pq_serial_range=[0,0,244]` means from pq00_000 to pq00_244."
            )
        },
    )
    use_wandb: Optional[bool] = field(
        default=False,
        metadata={"help": "If false, the program will not run W&B."},
    )


class Logger:
    def __init__(self, *args, **kwargs):
        self.run = None
        conf_dict = kwargs.get("config", None)
        if conf_dict and conf_dict["use_wandb"]:
            self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        if self.run:
            wandb.log(dic)

    def close(self):
        if self.run:
            wandb.finish()


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


# Replicated from ~`bsmetadata.train.show_help``to avoid the import.
def show_help(context="", cls=PreprocessingConfig):
    default_instance = cls()
    for field_ in fields(cls):
        if is_dataclass(field_.type):
            show_help(context=f"{context}{field_.name}.", cls=field_.type)
        else:
            kwargs = field_.metadata.copy()
            help = kwargs.get("help", "")
            default = getattr(default_instance, field_.name)  # init and tell the default
            print(f"{context}{field_.name}: {help} (default={json.dumps(default)})")


col_html = "html"
col_url = "url"
col_to_store_text = "text"
col_to_store_head = "html_head"
col_to_store_footer = "html_footer"
col_to_store_title = "html_title"
col_to_store_metadata_html = "metadata_html"
col_to_store_metadata_url = "metadata_url"
col_to_store_metadata_timestamp = "metadata_timestamp"
col_to_store_metadata_title = "metadata_title"
col_to_store_metadata_website_desc = "metadata_website_desc"
col_to_store_metadata_entities = "metadata_entity"
col_to_store_metadata_entity_paragraph = "metadata_entity_paragraph"
col_to_store_metadata_generation_length_text = "metadata_generation_length_text"
col_to_store_metadata_generation_length_sentence = "metadata_generation_length_sentence"
col_to_store_metadata_datasource = "metadata_generation_datasource"
col_to_store_metadata_paragraph = "metadata_paragraph"


@hydra.main(config_path="../../../jz/dataset/c4/python_scripts", config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format=(
            "[<blue>{time:YYYY-MM-DD HH:mm:ss.SSS}</blue>][<level>{level: >8}</level>][<green>{name}</green>]"
            " - <level>{message}</level>"
        ),
        colorize=True,
    )

    config_dict = OmegaConf.to_container(args)
    metrics_logger = Logger(project=args.project_name, config=config_dict)

    logger.info("Initialize the preprocessors:")
    if "html" in args.metadata_to_include:
        logger.info("   Html...")
        _html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html,
            col_to_store_text=col_to_store_text,
            col_html=col_html,
            col_to_store_footer=col_to_store_footer,
            col_to_store_head=col_to_store_head,
            col_to_store_title=col_to_store_title,
        )
        html_processor = ErrorWrapperPreprocessor(
            metadata_preprocessor=_html_processor,
            output_keys={
                col_to_store_metadata_html: [],
                col_to_store_text: "",
                col_to_store_footer: [],
                col_to_store_head: [],
                col_to_store_title: [],
            },
        )

    if "url" in args.metadata_to_include:
        logger.info("   Url...")
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url=col_url)

    if "timestamp" in args.metadata_to_include:
        logger.info("   Timestamp...")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )

    if "title" in args.metadata_to_include:
        logger.info("   Title...")
        title_processor = TitlePreprocessor(
            col_to_store_metadata=col_to_store_metadata_title, col_title=col_to_store_title
        )

    if "website_description" in args.metadata_to_include:
        logger.info("   Website description...")
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc,
            col_metadata_url=col_to_store_metadata_url,
            path_wiki_db=args.path_wiki_db,
        )

    if "clean_website_description" in args.metadata_to_include:
        logger.info("   Clean Website Description...")
        website_desc_cleaner = WebsiteDescPostprocessor(col_to_store_metadata=col_to_store_metadata_website_desc)

    if "entity" in args.metadata_to_include:
        logger.info("   Entity...")
        entity_processor = EntityPreprocessor(
            base_url=args.entity_path_data_dir,
            path_or_url_flair_ner_model=args.path_or_url_flair_ner_model,
            col_to_store_metadata=col_to_store_metadata_entities,
            col_text=col_to_store_text,
        )

    if "entity_paragraph" in args.metadata_to_include:
        logger.info("   Entity Paragraph...")
        entity_paragraph_processor = EntityParagraphPreprocessor(
            col_to_store_metadata=col_to_store_metadata_entity_paragraph,
            col_entity=col_to_store_metadata_entities,
            col_paragraph=col_to_store_metadata_paragraph,
        )

    if "generation_length_text" in args.metadata_to_include:
        logger.info("   Generation length text...")
        generation_length_preprocessor_text = GenerationLengthPreprocessor(
            mode="text", col_to_store_metadata=col_to_store_metadata_generation_length_text
        )

    if "generation_length_sentence" in args.metadata_to_include:
        logger.info("   Generation length sentence...")
        generation_length_preprocessor_sentence = GenerationLengthPreprocessor(
            mode="sentence", col_to_store_metadata=col_to_store_metadata_generation_length_sentence
        )

    if "datasource" in args.metadata_to_include:
        logger.info("   Datasource...")
        datasource_preprocessor = DatasourcePreprocessor(
            col_to_store_metadata=col_to_store_metadata_datasource, col_url="url"
        )

    if "paragraph" in args.metadata_to_include:
        logger.info("   Paragraph...")
        paragraph_preprocessor = ParagraphPreprocessor(col_to_store_metadata=col_to_store_metadata_paragraph)

    logger.info("Processors initialization finished")

    if args.set_dataset is not None:
        poss_files = [args.set_dataset]
    elif args.file_pq_serial_range:
        poss_files = [
            f"c4-en-html_cc-main-2019-18_pq{args.file_pq_serial_range[0]:02}-{i:03}.jsonl.gz"
            for i in range(args.file_pq_serial_range[1], args.file_pq_serial_range[2] + 1)
        ]
    else:
        poss_files = sorted(os.listdir(args.dataset_name))

    if args.use_load_from_disk:
        poss_files = [file_name for file_name in poss_files if file_name.startswith("c4-en-html")]
    else:
        poss_files = [
            file_name
            for file_name in poss_files
            if (file_name.endswith("jsonl.gz") or file_name.endswith("jsonl")) and file_name.startswith("c4-en-html")
        ]

    _CACHE_DIR = args.cache_dir or config.HF_DATASETS_CACHE
    logger.info(_CACHE_DIR)

    def process_file(file_name: str):
        if file_name.endswith(".jsonl.gz"):
            out_file_name = file_name[: -len(".jsonl.gz")]
        elif file_name.endswith(".jsonl"):
            out_file_name = file_name[: -len(".jsonl")]
        else:
            out_file_name = file_name
        out_file_name_tmp = f"tmp-{out_file_name}"

        saving_path = os.path.join(args.out_dir, out_file_name)
        saving_path_tmp = os.path.join(args.out_dir, out_file_name_tmp)

        if args.skip_if_save_file_already_exist:
            if os.path.isfile(os.path.join(saving_path, "dataset.arrow")):
                logger.warning(f"Skipping the processing of {file_name} as the saved processed dataset already exist")
                return

        processing_name = (
            "-".join(args.metadata_to_include) if args.metadata_to_include is not None else "full-process"
        )
        metrics_logger.log({processing_name: 0})

        metrics_logger.log({"load_dataset": 0})
        if args.use_load_from_disk:
            dataset_name = os.path.join(args.dataset_name, file_name)
            logger.info(f"Loading the dataset {dataset_name} with `load_from_disk`")
            ds = load_from_disk(dataset_name)
        else:
            data_files = {"file": file_name}
            logger.info(
                "Loading a dataset with `load_dataset`:\n"
                f"\t{args.dataset_name}, {args.dataset_config_name}\n"
                f"\trevision={args.dataset_revision}\n"
                f"\tdata_files={data_files}\n"
                f"\tcache_dir={args.cache_dir}"
            )

            def _try_to_get_ds() -> Dataset:
                # For `load_dataset()` to cast `null` of `metadata_timestamp`.
                _feature_dic = {
                    "c4_shard": Value(dtype="int64", id=None),
                    "c4_timestamp": Value(dtype="string", id=None),
                    "html": Value(dtype="string", id=None),
                    "url": Value(dtype="string", id=None),
                    "metadata_html": [
                        {
                            "char_end_idx": Value(dtype="int64", id=None),
                            "char_start_idx": Value(dtype="int64", id=None),
                            "html_attrs": {
                                "attrs": [Value(dtype="string", id=None)],
                                "values": [Value(dtype="string", id=None)],
                            },
                            "key": Value(dtype="string", id=None),
                            "relative_end_pos": Value(dtype="int64", id=None),
                            "relative_start_pos": Value(dtype="int64", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "text": Value(dtype="string", id=None),
                    "html_footer": [Value(dtype="string", id=None)],
                    "html_head": [Value(dtype="string", id=None)],
                    "html_title": [Value(dtype="string", id=None)],
                    # See ~`bsmetadata.preprocessing_utils.ErrorWrapperPreprocessor.new_columns_minimal_features`
                    # for a `Value` vs. `List[Value]` situation.
                    # Not sure what happened.
                    "HtmlPreprocessor_error": Value(dtype="int64", id=None),
                    "HtmlPreprocessor_error_comment": Value(dtype="string", id=None),
                    "metadata_url": [
                        {
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_timestamp": [
                        {
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_generation_length_text": [
                        {
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_generation_length_sentence": [
                        {
                            "char_end_idx": Value(dtype="int64", id=None),
                            "char_start_idx": Value(dtype="int64", id=None),
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_generation_datasource": [
                        {
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_website_desc": [
                        {
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ],
                    "metadata_paragraph": [
                        {
                            "char_end_idx": Value("int64"),
                            "char_start_idx": Value("int64"),
                            "key": Value("string"),
                            "type": Value("string"),
                            "value": Value("string"),
                            "marker": Value("string"),
                        }
                    ],
                }
                _e: Union[KeyError, ValueError, Exception] = None
                try:
                    # For unknown reasons, this occationally hang after the HF's message
                    # "Extracting data files: 100%" without any error.
                    # Restarting the script always fixes it, mysteriously.
                    return load_dataset(
                        args.dataset_name,
                        args.dataset_config_name,
                        data_files=data_files,
                        features=Features(_feature_dic),
                        cache_dir=args.cache_dir,
                        keep_in_memory=False,
                        download_mode=args.download_mode,
                        revision=args.dataset_revision,
                        use_auth_token=True,
                    )["file"]
                except (KeyError, ValueError, Exception) as e:
                    _e = e
                    logger.warning(e.__class__.__name__ + " (`Features` need metadata_entity)")
                    pass
                try:
                    _feature_dic["metadata_entity"] = [
                        {
                            "char_end_idx": Value(dtype="int64", id=None),
                            "char_start_idx": Value(dtype="int64", id=None),
                            "key": Value(dtype="string", id=None),
                            "type": Value(dtype="string", id=None),
                            "value": Value(dtype="string", id=None),
                        }
                    ]
                    return load_dataset(
                        args.dataset_name,
                        args.dataset_config_name,
                        data_files=data_files,
                        features=Features(_feature_dic),
                        cache_dir=args.cache_dir,
                        keep_in_memory=False,
                        download_mode=args.download_mode,
                        revision=args.dataset_revision,
                        use_auth_token=True,
                    )["file"]
                except (KeyError, ValueError, Exception) as e:
                    logger.exception(e)
                    raise e from _e

            ds = _try_to_get_ds()

        metrics_logger.log({"load_dataset": 1})

        if args.select_n_first_indices:
            logger.info(f"Extract the first-{args.select_n_first_indices} indices from the dataset")
            ds = ds.select([i for i in range(args.select_n_first_indices)])

        features_dict = dict(ds.features)
        logger.info(f"The initial features of the dataset are:\n{pformat(list(features_dict.keys()), compact=True)}")

        def apply_taggers(ds: Dataset, taggers: List[MetadataTagger]) -> Dataset:
            tagger_names = []
            for tagger in taggers:
                for col_name, feature_type in tagger.new_columns_minimal_features.items():
                    # assert col_name not in features_dict
                    features_dict[col_name] = feature_type
                tagger_names.append(tagger.__class__.__name__)
            chained_funcs = partial(lambda x: reduce(lambda g, f: f(g), [tagger.tag for tagger in taggers], x))

            tagger_pipe_str = "|".join(tagger_names)
            logger.info(f"Start {tagger_pipe_str}")
            metrics_logger.log({tagger_pipe_str: 0})
            logger.info(f"   Example of 1st example 100 first characters:\n\t{repr(ds[0][col_to_store_text][:100])}")
            ds = ds.map(
                chained_funcs,
                batched=True,
                batch_size=args.map_batch_size,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=tagger_pipe_str,
                features=Features(features_dict),
            )
            metrics_logger.log({tagger_pipe_str: 1})
            logger.info(f"End {tagger_pipe_str}")
            return ds

        tagger_list = []
        if "html" in args.metadata_to_include:
            tagger_list.append(html_processor)

        if "url" in args.metadata_to_include:
            tagger_list.append(url_processor)

        if "timestamp" in args.metadata_to_include:
            tagger_list.append(timestamp_processor)

        if "title" in args.metadata_to_include:
            tagger_list.append(title_processor)

        if "website_description" in args.metadata_to_include:
            tagger_list.append(website_processor)

        if "clean_website_description" in args.metadata_to_include:
            tagger_list.append(website_desc_cleaner)

        if "entity" in args.metadata_to_include:
            tagger_list.append(entity_processor)

        has_entity = "metadata_entity" in features_dict
        logger.debug(f"has_entity={has_entity}")
        if "entity_paragraph" in args.metadata_to_include and has_entity:
            tagger_list.append(entity_paragraph_processor)

        if "generation_length_text" in args.metadata_to_include:
            tagger_list.append(generation_length_preprocessor_text)

        if "generation_length_sentence" in args.metadata_to_include:
            tagger_list.append(generation_length_preprocessor_sentence)

        if "datasource" in args.metadata_to_include:
            tagger_list.append(datasource_preprocessor)

        if "paragraph" in args.metadata_to_include:
            tagger_list.append(paragraph_preprocessor)

        ds = apply_taggers(ds, tagger_list)

        logger.info(
            "Saving the processed dataset:\n"
            f"{pformat({'features': ds.column_names, 'num_rows': ds.num_rows}, compact=True)}"
        )
        metrics_logger.log({"save_result": 0})
        if not args.save_as_jsonl_gz:
            ds.save_to_disk(saving_path_tmp)
        else:
            saving_path_tmp += ".jsonl.gz"
            saving_path += ".jsonl.gz"
            # Without a swapfile, some big records need 20GB RAM for `save_as_jsonl_gz_num_workers > 1`
            ds.to_json(
                saving_path_tmp,
                batch_size=args.save_as_jsonl_gz_batch_size,
                num_proc=args.save_as_jsonl_gz_num_workers,
                compression="gzip",
            )
        logger.info(f"   Saved as {os.path.abspath(saving_path_tmp)}")
        logger.info(f"Moving {os.path.abspath(saving_path_tmp)} to {os.path.abspath(saving_path)}")
        shutil.move(saving_path_tmp, saving_path)
        logger.info(f"Processing of {file_name} ended successfully")
        metrics_logger.log({processing_name: 1})

        removal_cnt = ds.cleanup_cache_files()
        logger.info(f"{removal_cnt} cache files is removed")
        shutil.rmtree(_CACHE_DIR)
        logger.info(f"{_CACHE_DIR} is removed")

    if not args.file_pq_serial_range and args.set_dataset is None:
        poss_files = poss_files[
            args.task_id * args.num_files_to_process : args.task_id * args.num_files_to_process
            + args.num_files_to_process
        ]
    for file_name in poss_files:
        logger.info(f"Start to process {file_name}")
        process_file(file_name=file_name)

    metrics_logger.close()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
