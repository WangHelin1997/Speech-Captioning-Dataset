CACHE_DIR="/content/cache/out/gigaspeech-tiny-0-train"
OUTPUT_DIR="/content/cache/out/gigaspeech-tiny-0-train-stage1"

CUDA_VISIBLE_DEVICES=0 python main.py \
  --cache_dir ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --batch_size 8 \
  --rename_column \
  --apply_squim_quality_estimation 
