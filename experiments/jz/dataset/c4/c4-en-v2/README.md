These are scripts to test the extraction pipeline on some files that should be identical to the dataset files that will be provided by Mike.

The extraction pipeline will be executed in 3 or 4 or more steps:
1. extraction of text, html, url, generation length text, generation lengh sentence and timestamp. After this step the dataset will have the columns: `['c4_shard', 'c4_timestamp', 'url', 'metadata_html', 'text', 'html_footer', 'html_head', 'html_title', 'HtmlPreprocessor_error', 'HtmlPreprocessor_error_comment', 'metadata_url', 'metadata_timestamp', 'metadata_generation_length_text', 'metadata_generation_length_sentence', 'metadata_generation_datasource']`
2. extraction of web site descriptions
3. extraction of entities
4. dataset cleanup: remove (and count) the "empty" rows, remove the useless columns like 'HtmlPreprocessor_error' or 'HtmlPreprocessor_error_comment',, and if needed put back in a single column all the metadata that are currently separated in several columns.

Current status: 

- `01_add_metadata_to_toy_c4_dataset.slurm` should tale 2_000h to process 3_000 files
- `02_add_website_desc.slurm` should tale 500h to process 3_000 files (but I think that some files will cause some issues)
- `03_add_entities.slurm`not working currently