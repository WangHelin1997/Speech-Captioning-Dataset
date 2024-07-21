CACHE_DIR="/content/cache/out/gigaspeech-tiny-0-train"
OUTPUT_DIR="/content/cache/out/gigaspeech-tiny-0-train-stage1"

CUDA_VISIBLE_DEVICES=0 python main_split.py \
  --cache_dir ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --batch_size 8 \
  --rename_column \
  --apply_squim_quality_estimation 


CACHE_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-0-train-stage1"
OUTPUT_DIR="/data/lmorove1/hwang258/sc/Speech-Captioning-Dataset/cache/out/gigaspeech-tiny-0-train-stage2"
CSV_DIR="/data/lmorove1/hwang258/dataspeech/cache" 

python annotate_split.py \
  --cache_dir ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --csv_dir ${CSV_DIR}
