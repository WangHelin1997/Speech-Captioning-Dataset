CACHE_DIR="/data/lmorove1/hwang258/dataspeech/cache"
OUTPUT_DIR="/data/lmorove1/hwang258/dataspeech/output"

python download.py # change cache path

CUDA_VISIBLE_DEVICES=0 python main.py "westbrook/gigaspeech-xl" \
  --configuration "default" \
  --cache_dir ${CACHE_DIR} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 16 \
  --batch_size 16 \
  --rename_column \
  --apply_squim_quality_estimation \
  --repo_id "gigaspeech-tiny-stage1"

python annotate.py "westbrook/gigaspeech-xl-stage1" \
  --configuration "default" \
  --cache_dir ${CACHE_DIR} \
  --repo_id "gigaspeech-tiny-stage2" \
  --csv_dir "/data/lmorove1/hwang258/dataspeech/cache"

python ./scripts/metadata_to_text.py \
    "westbrook/gigaspeech-xl-stage2" \
    --repo_id "gigaspeech-xl-stage3" \
    --cache_dir ${CACHE_DIR} \
    --configuration "default" \
    --cpu_num_workers 16

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-xl-stage3" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 8 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-xl-stage4"