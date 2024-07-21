export CUDA_VISIBLE_DEVICES=0
REPO="gigaspeech-tiny-train"
CONFIG='default'
CSV_DIR="/data/lmorove1/hwang258/dataspeech/cache" 
MODEL_CACHE_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/model/"
CACHE_DIRS="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/"
OUTPUT_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train"

N=0
CACHE_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N"
CACHE_DIR1="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage1"
CACHE_DIR2="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage2"
CACHE_DIR3="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage3"
CACHE_DIR4="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage4"
CACHE_DIR5="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage5"
CACHE_DIR6="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage6"
CACHE_DIR7="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage7"
CACHE_DIR8="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-stage8"
CACHE_BIN_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-train-$N-bin"

# annotate speed, pitch, ...
python main_split.py \
  --cache_dir ${CACHE_DIR} \
  --output_dir ${CACHE_DIR1} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --batch_size 8 \
  --rename_column \
  --apply_squim_quality_estimation 

# annotate age, gender, ...
python annotate_split.py \
  --cache_dir ${CACHE_DIR1} \
  --output_dir ${CACHE_DIR2} \
  --csv_dir ${CSV_DIR}

# convert speed, pitch, ... to text
python ./scripts/metadata_to_text_split.py \
    --cache_dir ${CACHE_DIR2} \
    --output_dir ${CACHE_DIR3} \
    --save_bin_edges ${CACHE_BIN_DIR} \
    --cpu_num_workers 8
    
# generate 1st description
python ./scripts/run_prompt_creation_split.py \
  --cache_dir ${MODEL_CACHE_DIR} \
  --dataset_cache_dir ${CACHE_DIR3} \
  --output_dir ${CACHE_DIR4} \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --load_in_4bit \
  --num_description 1

# generate 2nd description
python ./scripts/run_prompt_creation_split.py \
  --cache_dir ${MODEL_CACHE_DIR} \
  --dataset_cache_dir ${CACHE_DIR4} \
  --output_dir ${CACHE_DIR5} \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --load_in_4bit \
  --num_description 2

# generate 3rd description
python ./scripts/run_prompt_creation_split.py \
  --cache_dir ${MODEL_CACHE_DIR} \
  --dataset_cache_dir ${CACHE_DIR5} \
  --output_dir ${CACHE_DIR6} \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --load_in_4bit \
  --num_description 3

# generate 4th description
python ./scripts/run_prompt_creation_split.py \
  --cache_dir ${MODEL_CACHE_DIR} \
  --dataset_cache_dir ${CACHE_DIR6} \
  --output_dir ${CACHE_DIR7} \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --load_in_4bit \
  --num_description 4

# generate 5th description
python ./scripts/run_prompt_creation_split.py \
  --cache_dir ${MODEL_CACHE_DIR} \
  --dataset_cache_dir ${CACHE_DIR7} \
  --output_dir ${CACHE_DIR8} \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --load_in_4bit \
  --num_description 5
