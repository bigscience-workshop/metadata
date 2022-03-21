#!/bin/bash
#SBATCH --job-name=add-entities-batch-1-2-retry-2-mod-meta-bs         # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=4                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 20:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/add-entities-test-retry-oom/%j-%x.out  
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=1356,1547,1530,298,366,131,1409,966,2366,1961,1512,1616,1011,734,133,986,1654,982,825,2468,1726,355,2503,1044,889,2347,2485,291,1514,2076,1960,1405,2078,981,1401,1156,1549,1968,299,1276,972,1352,304,2073,697,33,504,2079,1599,1805,973,569,980,2208,301,1033,1031,196,946,1540,605,1511,954,501,956,1618,988,772,1807,282,2502,403,1097,1157,726,1966,1261,2310,1969,1154,631,387,748,955,1402,1544,1545,1967,1936,1972,1670,240,83,1012,1546,1635,1655,1962,1964,683,444,244,1513,1035,989,2114,1519,576,574,1560,1153,78,10,1010,2293,1937,1350,604,163,406,1482,1192,2071,2458,950,292,1889,1400,1542,400,1032,300,135,309,1965,2389,2435,401,831,1039,339,977,2075,2077,2070,778,959,1277,18,2434,1693,2430,1155,1548,1323,1460,1543,410,500,1274,2183,69,1526,2,1746,832,407,1963,1275,164,2431,113,2072,941,2432,985,1510,1751,984,1464,2001,1806,296,1515,402,1086,983,2344,1873,226,1615,2356,1236,979,317,1541,2437,293,24,947,16
#SBATCH --partition=cpu_p1

set -x -e

# Next line will:
# - load a conda environment with the dependencies on the master branch of github.com/bigscience-workshop/metadata/
# - setup env vars ($HOME, $WORK, etc)
# - load several modules (git)
# Note: We can afford to have only two conda environments: one stable for running experiments and one for development.
# If there are new dependencies to install, you have to tell me about them and not do it in this script
source $HOME/start-modelling-metadata-user

cd $WORK/repos/metadata/

# We are on an offline partition
export FLAIR_CACHE_ROOT=$SCRATCH/cache_dir/flair
export HF_DATASETS_OFFLINE=1
# be careful about the cache folder for Wandb
export WANDB_MODE=offline
export WANDB_DIR=$SCRATCH
export HF_DATASETS_CACHE=$SCRATCH/to_delete

METADATA_TO_INCLUDE='["entity"]'

PROCESS_DIR=/gpfsscratch/rech/six/uue59kq/new_dataset
DATASET_FILES_DIR=$PROCESS_DIR/c4-en-html-with-metadata-sharded
OUT_DIR=$PROCESS_DIR/process-v2/c4-en-sharded-with-entity

mkdir -p $OUT_DIR

MAP_BATCH_SIZE=1
PREPROCESSING_NUM_WORKERS=1
NUM_FILES_TO_PROCESS=1
SAVE_BATCH_SIZE=100

PATH_OR_URL_FLAIR_NER_MODEL=$SCRATCH/cache_dir/flair/ner-fast/en-ner-fast-conll03-v0.4.pt

echo "Args:"
echo "    task_id=${SLURM_ARRAY_TASK_ID}"
echo "    map_batch_size=$MAP_BATCH_SIZE"
echo "    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS"
echo "    out_dir=$OUT_DIR"
echo "    dataset_name=$DATASET_FILES_DIR"
echo "    metadata_to_include=$METADATA_TO_INCLUDE"
echo "    path_or_url_flair_ner_model=$PATH_OR_URL_FLAIR_NER_MODEL"

python experiments/jz/dataset/c4/python_scripts/add_metadata.py \
    task_id=${SLURM_ARRAY_TASK_ID}\
    out_dir=$OUT_DIR \
    path_wiki_db=$SCRATCH/modeling-metadata-artefacts/wiki_en_dump.db \
    entity_path_data_dir=$SCRATCH/modeling-metadata-artefacts/entity_preprocessing \
    dataset_name=$DATASET_FILES_DIR \
    metadata_to_include="$METADATA_TO_INCLUDE" \
    path_or_url_flair_ner_model=$PATH_OR_URL_FLAIR_NER_MODEL \
    map_batch_size=$MAP_BATCH_SIZE \
    preprocessing_num_workers=$PREPROCESSING_NUM_WORKERS\
    num_files_to_process=$NUM_FILES_TO_PROCESS\
    save_batch_size=$SAVE_BATCH_SIZE\
    use_load_from_disk=true\
    skip_if_save_file_already_exist=true
