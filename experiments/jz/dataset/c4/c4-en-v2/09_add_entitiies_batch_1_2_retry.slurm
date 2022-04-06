#!/bin/bash
#SBATCH --job-name=add-entities-batch-1-2-retry-mod-meta-bs         # (change me!) job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                                             # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=2                                               # (change me! between 0 and 48) number of cores per tasks
#SBATCH --hint=nomultithread                                            # we get physical cores not logical
#SBATCH --time 20:00:00                                                 # (change me! between 0 and 20h) maximum execution time (HH:MM:SS)
#SBATCH --output=/gpfsdswork/projects/rech/six/uue59kq/logs/modelling-metadata/add-entities-test-retry/%j-%x.out  
#SBATCH --account=six@cpu                                               # account
#SBATCH --array=924,2302,1955,428,1120,482,351,2084,928,344,165,1996,350,2074,938,272,2009,1145,1065,1069,836,1096,911,2272,803,1064,16,1139,1137,2007,939,425,29,218,2087,205,167,1178,352,212,1095,28,1059,909,1052,208,285,1068,1090,1923,380,2298,2011,1141,1054,1984,1158,267,842,1067,823,1920,1182,510,424,375,360,1050,970,2285,1103,370,1172,216,2175,316,268,2018,1099,845,2283,841,929,369,849,158,2021,312,1181,1113,1943,727,1098,377,925,203,804,1066,796,1956,262,2056,820,1060,919,269,1148,1055,1186,1140,261,1952,1942,2290,913,927,492,307,1929,1941,217,1993,1992,616,1134,2318,1104,361,1951,2303,2066,821,1122,844,1167,1114,222,1058,347,1091,484,206,219,2014,2267,918,1117,565,2170,2088,987,1118,2057,2321,2495,791,935,55,416,838,270,2300,1061,1958,321,2027,25,1094,931,1173,1062,2291,2297,937,828,2186,1136,378,2207,930,933,1957,1063,363,1051,1177,1053,426,1092,258,2230,2234,1056,346,2490,146,1183,354,2269,1980,214,1160,1087,427,359,412,910,1169,348,1184,795,263,1168,1166,24,990,872,1187,414,2282,2016,1057,793,936,1115,2174,423,2284,421,1138,807,213,2022,1174,1171,1107,1170,867,497,1121,461,1125,471,1176,2006,499,932,1106,2179,1945,1180,1147,1185,1179,2315,2082,30,356,1188,2012,349,1093,371,1162,2294,934,211,147,413,719,2206,73,1997,220,479,2020,1116,1189,2086,172,1924
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
