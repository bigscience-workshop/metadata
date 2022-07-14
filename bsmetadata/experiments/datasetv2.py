from fnmatch import fnmatch

import datasets
from datasets import Features, concatenate_datasets, interleave_datasets, load_dataset
from datasets.filesystems import HfFileSystem
from huggingface_hub import dataset_info


data_files_with_entities = [
    "c4-en-html_cc-main-2019-18_pq00-000.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-001.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-002.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-003.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-004.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-005.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-006.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-007.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-008.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-009.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-010.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-011.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-012.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-013.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-014.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-015.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-016.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-017.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-018.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-019.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-020.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-021.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-022.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-023.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-024.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-025.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-026.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-027.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-028.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-029.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-030.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-031.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-032.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-033.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-034.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-035.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-036.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-037.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-038.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-039.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-040.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-041.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-042.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-043.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-044.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-045.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-046.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-047.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-048.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-049.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-050.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-051.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-052.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-053.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-054.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-055.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-056.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-057.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-058.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-059.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-060.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-061.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-062.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-063.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-064.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-065.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-066.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-067.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-068.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-069.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-070.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-071.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-072.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-073.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-074.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-075.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-076.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-077.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-078.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-079.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-080.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-081.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-082.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-083.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-084.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-085.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-086.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-087.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-088.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-089.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-090.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-091.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-092.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-093.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-094.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-095.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-096.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-097.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-098.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-099.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-100.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-101.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-102.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-103.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-104.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-105.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-106.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-107.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-108.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-109.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-110.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-111.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-112.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-113.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-114.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-115.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-116.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-117.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-118.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-120.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-121.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-122.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-123.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-124.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-125.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-126.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-127.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-128.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-129.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-130.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-131.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-132.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-133.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-134.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-135.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-136.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-137.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-138.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-139.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-140.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-141.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-142.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-143.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-144.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-145.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-146.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-147.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-148.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-149.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-150.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-151.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-152.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-153.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-154.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-155.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-156.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-157.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-158.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-159.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-160.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-161.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-162.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-163.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-164.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-165.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-166.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-167.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-168.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-169.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-170.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-171.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-172.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-173.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-174.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-175.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-176.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-177.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-178.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-179.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-180.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-181.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-182.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-183.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-184.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-185.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-186.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-187.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-188.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-189.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-190.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-191.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-192.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-193.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-194.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-195.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-196.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-197.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-198.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-199.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-200.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-201.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-202.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-203.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-204.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-205.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-206.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-207.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-208.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-209.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-210.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-211.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-212.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-213.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-214.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-215.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-216.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-217.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-218.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-219.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-220.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-221.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-222.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-223.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-224.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-225.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-226.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-227.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-228.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-229.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-230.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-231.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-232.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-233.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-234.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-235.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-236.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-237.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-238.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-239.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-240.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-241.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-242.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-243.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq00-244.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-000.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-001.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-002.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-003.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-004.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-005.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-006.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-007.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-008.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-009.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-010.jsonl.gz",
    "c4-en-html_cc-main-2019-18_pq01-011.jsonl.gz",
]

features = {
    "HtmlPreprocessor_error": {"_type": "Value", "dtype": "int64", "id": None},
    "HtmlPreprocessor_error_comment": {"_type": "Value", "dtype": "string", "id": None},
    "c4_shard": {"_type": "Value", "dtype": "int64", "id": None},
    "c4_timestamp": {"_type": "Value", "dtype": "string", "id": None},
    "html": {"_type": "Value", "dtype": "string", "id": None},
    "html_footer": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_head": [{"dtype": "string", "id": None, "_type": "Value"}],
    "html_title": [{"dtype": "string", "id": None, "_type": "Value"}],
    "metadata_generation_datasource": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_generation_length_sentence": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_generation_length_text": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_html": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "html_attrs": {
                "attrs": [{"_type": "Value", "dtype": "string", "id": None}],
                "values": [{"_type": "Value", "dtype": "string", "id": None}],
            },
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "relative_end_pos": {"_type": "Value", "dtype": "int64", "id": None},
            "relative_start_pos": {"_type": "Value", "dtype": "int64", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_paragraph": [
        {
            "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "marker": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_timestamp": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "timestamp[s]", "id": None},
        }
    ],
    "metadata_title": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_url": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "metadata_website_desc": [
        {
            "key": {"_type": "Value", "dtype": "string", "id": None},
            "type": {"_type": "Value", "dtype": "string", "id": None},
            "value": {"_type": "Value", "dtype": "string", "id": None},
        }
    ],
    "text": {"_type": "Value", "dtype": "string", "id": None},
    "url": {"_type": "Value", "dtype": "string", "id": None},
}
features_with_entities = features.copy()
features_with_entities.update(
    {
        "metadata_entity": [
            {
                "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "key": {"_type": "Value", "dtype": "string", "id": None},
                "type": {"_type": "Value", "dtype": "string", "id": None},
                "value": {"_type": "Value", "dtype": "string", "id": None},
            }
        ],
        "metadata_entity_paragraph": [
            {
                "char_end_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "char_start_idx": {"_type": "Value", "dtype": "int64", "id": None},
                "key": {"_type": "Value", "dtype": "string", "id": None},
                "relative_end_pos": {"_type": "Value", "dtype": "int64", "id": None},
                "relative_start_pos": {"_type": "Value", "dtype": "int64", "id": None},
                "type": {"_type": "Value", "dtype": "string", "id": None},
                "value": {"_type": "Value", "dtype": "string", "id": None},
            }
        ],
    }
)


def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        try:
            return getattr(datasets, features["_type"])(features["dtype"])
        except ValueError:
            print(features)
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]


new_features = {}
final_features = convert_types(features)
final_features_with_entities = convert_types(features_with_entities)

di = dataset_info("bs-modeling-metadata/c4-en-html-with-metadata")
fs = HfFileSystem(di)
all_files = fs.ls(".")


def get_files(pattern):
    for file in all_files:
        if fnmatch(file, pattern):
            yield file


def load_dataset_by_files(files, streaming=False):
    selected_files_entities = list(filter(lambda v: v in data_files_with_entities, files))
    selected_files_no_entities = list(filter(lambda v: v not in data_files_with_entities, files))
    datasets = []
    if selected_files_entities:
        dataset_entities = load_dataset(
            "bs-modeling-metadata/c4-en-html-with-metadata",
            features=Features(final_features_with_entities),
            data_files=selected_files_entities,
            split="train",
            use_auth_token=True,
            streaming=streaming,
        )
        datasets.append((dataset_entities, len(selected_files_entities)))

    if selected_files_no_entities:
        dataset_no_entities = load_dataset(
            "bs-modeling-metadata/c4-en-html-with-metadata",
            features=Features(final_features),
            data_files=selected_files_no_entities,
            split="train",
            use_auth_token=True,
            streaming=streaming,
        )
        datasets.append((dataset_no_entities, len(selected_files_no_entities)))
    if not streaming:
        dataset = concatenate_datasets([d for d, _ in datasets])
    else:
        datasets = [d.shuffle(seed=42, buffer_size=1024) for d, n in datasets]
        if len(datasets) == 1:
            return datasets[0]
        sizes = [n for _, n in datasets]
        probabilities = [n / sum(sizes) for n in sizes]
        dataset = interleave_datasets(datasets, probabilities=probabilities)
    return dataset
