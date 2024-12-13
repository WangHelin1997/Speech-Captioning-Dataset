from datasets import load_from_disk
from huggingface_hub import create_repo, login
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val/", type=str)
    parser.add_argument("--targetdir", default="/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val_cleaned/", type=str)
    # parser.add_argument("--repo", default="OpenSound/GigaCaps_val_2", type=str)
    args = parser.parse_args()

    datadir = args.datadir
    targetdir = args.targetdir
    dataset = load_from_disk(datadir)
    modified_dataset = dataset.remove_columns(['speaker', 'audio', 'url', 'category', 'phonemes', 'accent', 'brightness', 'smoothness', 'emotion'])

    # def filter_by_duration(example):
    #     duration = float(example['end_time'] - example['begin_time'])
    #     return duration >= 3  # Keep rows with duration >= 3 seconds

    # modified_dataset = modified_dataset.filter(filter_by_duration)
    modified_dataset.save_to_disk(targetdir)
