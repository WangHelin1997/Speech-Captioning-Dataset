from datasets import load_dataset, Dataset, DatasetDict
import os
from tqdm import tqdm
import soundfile as sf

cache_dir = '/content/cache'
save_dir = os.path.join(cache_dir, 'audios')
out_dir = '/content/cache/out'

# Load the dataset
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)

original_dataset = load_dataset("westbrook/gigaspeech-tiny", "default", cache_dir=cache_dir)

N = 5
for split in original_dataset.keys():
  num_samples = len(original_dataset[split])
  num_per_subset = num_samples // N

  for i in range(N):
    start_idx = num_per_subset * i
    end_idx = num_per_subset * (i + 1) if i != N - 1 else num_samples
    split_dataset = original_dataset[split].select(range(start_idx, end_idx))
    split_dataset.push_to_hub(f'westbrook/gigaspeech-tiny-{i}', split)
    os.makedirs(os.path.join(out_dir,f'gigaspeech-tiny-{i}-{split}'), exist_ok=True)
    split_dataset.save_to_disk(os.path.join(out_dir,f'gigaspeech-tiny-{i}-{split}'))

## to read
# from datasets import load_from_disk

# loaded_new_dataset = load_from_disk('/content/cache/out/gigaspeech-tiny-0-train')
# print(loaded_new_dataset)
