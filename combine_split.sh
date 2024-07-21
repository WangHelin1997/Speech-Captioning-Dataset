REPO="gigaspeech-tiny-train"
CONFIG='default'
CACHE_DIRS="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/"
OUTPUT_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train"

# combine dataset and push to huggingface
python push_repo_split.py \
    --cache_dir ${CACHE_DIRS} \
    --output_dir ${OUTPUT_DIR} \
    --repo_id ${REPO} \
    --configuration ${CONFIG} 
    
