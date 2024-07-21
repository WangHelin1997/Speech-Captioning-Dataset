from datasets import concatenate_datasets, load_from_disk
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", default=None, type=str, help="Cache dir to download data")
parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
args = parser.parse_args()

dirs = os.listdir(args.cache_dir)
dataset_paths = [os.path.join(args.cache_dir, item)  for item in dirs if 'stage8' in item]

datasets = [load_from_disk(path) for path in dataset_paths]
combined_dataset = concatenate_datasets(datasets)

combined_dataset.save_to_disk(args.output_dir)
combined_dataset.push_to_hub(args.repo_id, args.configuration)

