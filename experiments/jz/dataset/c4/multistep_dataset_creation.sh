DATASET_FILES_DIR=$SCRATCH/new_dataset/hub/c4-en-reduced

PROCESS_DIR=$SCRATCH/new_dataset/c4-en-reduced-in-progess
OUT_DIR_1=$PROCESS_DIR/c4-en-reduced-with-metadata-timestamp
OUT_DIR_2=$PROCESS_DIR/c4-en-reduced-with-metadata-timestamp-website_description
OUT_DIR_3=$PROCESS_DIR/c4-en-reduced-with-metadata-timestamp-website_description-entity

mkdir $PROCESS_DIR
mkdir $OUT_DIR_1
mkdir $OUT_DIR_2
mkdir $OUT_DIR_3

PATH_OR_URL_FLAIR_NER_MODEL=$SCRATCH/cache_dir/flair/ner-fast/en-ner-fast-conll03-v0.4.pt

for filename in $DATASET_FILES_DIR/*; do
    FILENAME="${filename##*/}"
    FILENAME_END="${FILENAME: -4}" # test if this is a lock file
    NEW_FILENAME="${FILENAME::-3}l" #   the last 3 characters correspond to the compression algorithm format .gz

    if [ $FILENAME_END != "lock" ]; then    
        # Create job to add timestamp metadata
        METADATA_TO_INCLUDE="['timestamp']"
        DATASET_FILES_DIR=$SCRATCH/new_dataset/hub/c4-en-reduced
        OUT_DIR=$OUT_DIR_1

        # ID_JOB1=$(sbatch --job-name=modelling-metadata-c4-dataset-toy-add-metadata-timestamp \
        # --export=ALL,FILENAME=$FILENAME,NEW_FILENAME=$NEW_FILENAME,OUT_DIR=$OUT_DIR,METADATA_TO_INCLUDE=$METADATA_TO_INCLUDE,DATASET_FILES_DIR=$DATASET_FILES_DIR,PATH_OR_URL_FLAIR_NER_MODEL=$PATH_OR_URL_FLAIR_NER_MODEL \
        # 02_add_metadata_to_toy_c4_dataset.slurm | cut -d " " -f 4)

        # echo "Launch jobid $ID_JOB1 to extract timestamp from $DATASET_FILES_DIR/$FILENAME and save to $OUT_DIR/$NEW_FILENAME"

        # Create job to add website_description metadata
        METADATA_TO_INCLUDE="['website_description']"
        DATASET_FILES_DIR=$OUT_DIR
        OUT_DIR=$OUT_DIR_2

        NEW_FILENAME="$NEW_FILENAME.gz"

        # ID_JOB2=$(sbatch --dependency=afterok:$ID_JOB1 --job-name=modelling-metadata-c4-dataset-toy-add-metadata-website_description \
        # --export=ALL,FILENAME=$NEW_FILENAME,NEW_FILENAME=$NEW_FILENAME,OUT_DIR=$OUT_DIR,METADATA_TO_INCLUDE=$METADATA_TO_INCLUDE,DATASET_FILES_DIR=$DATASET_FILES_DIR,PATH_OR_URL_FLAIR_NER_MODEL=$PATH_OR_URL_FLAIR_NER_MODEL \
        # 02_add_metadata_to_toy_c4_dataset.slurm | cut -d " " -f 4)

        # echo "Launch jobid $ID_JOB2 to extract website_description from $DATASET_FILES_DIR/$FILENAME and save to $OUT_DIR/$NEW_FILENAME"

        # Create job to add entity metadata
        METADATA_TO_INCLUDE="['entity']"
        DATASET_FILES_DIR=$OUT_DIR
        OUT_DIR=$OUT_DIR_3

        ID_JOB3=$(sbatch --job-name=modelling-metadata-c4-dataset-toy-add-metadata-entity \
        --qos=qos_gpu-t3 \
        --account=six@gpu \
        --gres=gpu:1 \
        --constraint=v100-16g \
        --partition=gpu_p13 \
        --export=ALL,FILENAME=$NEW_FILENAME,NEW_FILENAME=$NEW_FILENAME,OUT_DIR=$OUT_DIR,METADATA_TO_INCLUDE=$METADATA_TO_INCLUDE,DATASET_FILES_DIR=$DATASET_FILES_DIR,PATH_OR_URL_FLAIR_NER_MODEL=$PATH_OR_URL_FLAIR_NER_MODEL \
        02_add_metadata_to_toy_c4_dataset.slurm | cut -d " " -f 4)

        echo "Launch jobid $ID_JOB3 to extract entity from $DATASET_FILES_DIR/$FILENAME and save to $OUT_DIR/$NEW_FILENAME"
    fi
done


