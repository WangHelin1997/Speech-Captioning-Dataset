from datasets import load_dataset, Dataset, DatasetDict
import os
from tqdm import tqdm
import soundfile as sf

cache_dir = '/data/lmorove1/hwang258/dataspeech/cache/'
save_dir = os.path.join(cache_dir, 'audios')

# Load the dataset
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(save_dir, exist_ok=True)
original_dataset = load_dataset("speechcolab/gigaspeech", "xl", num_proc=20, cache_dir=cache_dir)

# Extract the first 10 samples
# first_10_samples = original_dataset['train'].select(range(10))
# Save the first 10 samples to a new dataset
# small_dataset = DatasetDict({'train': first_10_samples})
# Push the new dataset to a repository on the Hugging Face Hub
# small_dataset.push_to_hub('westbrook/gigaspeech-tiny')

for idx, sample in tqdm(enumerate(original_dataset['train'])):
    audio = sample["audio"]

    # Extract the audio data and sampling rate
    audio_data = audio["array"]
    sampling_rate = audio["sampling_rate"]
    savename = sample["segment_id"]
    # Define the file path
    file_path = os.path.join(save_dir, f"{savename}.wav")

    # Save the audio file using soundfile
    sf.write(file_path, audio_data, sampling_rate)

print("All audio files saved successfully.")

