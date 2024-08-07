from datasets import load_dataset, Audio
from multiprocess import set_start_method
import argparse
import os
import re
import pandas as pd

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("dataset_name", type=str, help="Path or name of the dataset. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/loading_methods#datasets.load_dataset.path")
    parser.add_argument("--configuration", default=None, type=str, help="Dataset configuration to use, if necessary.")
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache dir to download data")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--csv_dir", default=None, type=str, help="Dir of csv files.")


    args = parser.parse_args()
    
    if args.configuration:
        dataset = load_dataset(args.dataset_name, args.configuration, num_proc=args.cpu_num_workers, cache_dir=args.cache_dir)
    else:
        dataset = load_dataset(args.dataset_name, num_proc=args.cpu_num_workers, cache_dir=args.cache_dir)

    segment_ids = [dataset['train'][i]['segment_id'] for i in range(len(dataset['train']))]
    print(segment_ids)

    csv_list = ['age','accent','brightness','emotion','gender','smoothness'] # add more here
    thres = 0.95
    for item in csv_list:
        # Load the CSV file
        df = pd.read_csv(os.path.join(args.csv_dir, item+'.csv'))
        df['segment_id'] = pd.Categorical(df['segment_id'], categories=segment_ids, ordered=True)
        sorted_df = df.sort_values(by='segment_id')
        dataset['train'] = dataset['train'].add_column(item+'_ori', list(sorted_df[item]))
        labeled = [item if value > thres else "None" for item, value in zip(list(sorted_df[item]), list(sorted_df[item+"_value"]))]
        dataset['train'] = dataset['train'].add_column(item+"_value", list(sorted_df[item+"_value"]))
        dataset['train'] = dataset['train'].add_column(item, labeled)

    # Function to update speaker name
    def update_speaker_column(example, index):
        new_speaker = f"TmpSpeaker_{index + 1}"
        return {'speaker': new_speaker}
    
    dataset['train'] = dataset['train'].map(update_speaker_column, with_indices=True)

    if args.output_dir:
        print("Saving to disk...")
        dataset.save_to_disk(args.output_dir)
    if args.repo_id:
        print("Pushing to the hub...")
        if args.configuration:
            dataset.push_to_hub(args.repo_id, args.configuration)
        else:
            dataset.push_to_hub(args.repo_id)
    
