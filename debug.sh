CACHE_DIR="/data/lmorove1/hwang258/dataspeech/cache"
OUTPUT_DIR="/data/lmorove1/hwang258/dataspeech/output"

CUDA_VISIBLE_DEVICES=0 python main.py "westbrook/gigaspeech-tiny" \
  --configuration "default" \
  --cache_dir ${CACHE_DIR} \
  --text_column_name "text" \
  --audio_column_name "audio" \
  --cpu_num_workers 8 \
  --batch_size 8 \
  --rename_column \
  --apply_squim_quality_estimation \
  --repo_id "gigaspeech-tiny-stage1"

python annotate.py "westbrook/gigaspeech-tiny-stage1" \
  --configuration "default" \
  --cache_dir ${CACHE_DIR} \
  --repo_id "gigaspeech-tiny-stage2" \
  --csv_dir "/data/lmorove1/hwang258/dataspeech/cache"

BIN_PATH="/data/lmorove1/hwang258/Speech-Captioning-Dataset/bin.json"

python ./scripts/metadata_to_text.py \
    "westbrook/gigaspeech-tiny-stage2" \
    --repo_id "gigaspeech-tiny-stage3" \
    --cache_dir ${CACHE_DIR} \
    --configuration "default" \
    --cpu_num_workers 8 \
    --path_to_bin_edges ${BIN_PATH}

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-tiny-stage3" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --overwrite_output_dir \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-tiny-stage4" \
  --num_description 1

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-tiny-stage4" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --overwrite_output_dir \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-tiny-stage4" \
  --num_description 2

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-tiny-stage4" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --overwrite_output_dir \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-tiny-stage4" \
  --num_description 3

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-tiny-stage4" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --overwrite_output_dir \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-tiny-stage4" \
  --num_description 4

python ./scripts/run_prompt_creation.py \
  --cache_dir ${CACHE_DIR} \
  --dataset_name "westbrook/gigaspeech-tiny-stage4" \
  --dataset_config_name "default" \
  --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.2" \
  --per_device_eval_batch_size 64 \
  --attn_implementation "sdpa" \
  --dataloader_num_workers 4 \
  --output_dir ${OUTPUT_DIR} \
  --load_in_4bit \
  --overwrite_output_dir \
  --push_to_hub \
  --hub_dataset_id "westbrook/gigaspeech-tiny-stage4" \
  --num_description 5
