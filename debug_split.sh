CACHE_DIR="/data/lmorove1/hwang258/dataspeech/cache"
OUTPUT_DIR="/data/lmorove1/hwang258/dataspeech/output"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --cache_dir ${CACHE_DIR} \
  --out_dir ${OUTPUT_DIR} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --batch_size 8 \
  --rename_column \
  --apply_squim_quality_estimation 
