from datasets import load_dataset, Audio, load_from_disk
from multiprocess import set_start_method
import argparse
import os
import re
import pandas as pd

if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
  
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache dir to download data")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--csv_dir", default=None, type=str, help="Dir of csv files.")


    args = parser.parse_args()
    
    dataset = load_from_disk(args.cache_dir)

    segment_ids = [dataset[i]['segment_id'] for i in range(len(dataset))]

    csv_list = ['age','accent','brightness','emotion','gender','smoothness'] # add more here
    thres = 0.95
    for item in csv_list:
        df = pd.read_csv(os.path.join(args.csv_dir, item + '.csv'))
        all_segments_df = pd.DataFrame({'segment_id': segment_ids})
        merged_df = all_segments_df.merge(df, on='segment_id', how='left')
        merged_df['segment_id'] = pd.Categorical(merged_df['segment_id'], categories=segment_ids, ordered=True)
        sorted_df = merged_df.sort_values(by='segment_id')
        dataset = dataset.add_column(item + '_ori', sorted_df[item].fillna("None").tolist())
        labeled = [item if value > thres else "None" for item, value in zip(sorted_df[item].fillna("None"), sorted_df[item + "_value"].fillna(0))]
        
        dataset = dataset.add_column(item + "_value", sorted_df[item + "_value"].fillna(0).tolist())
        dataset = dataset.add_column(item, labeled)

    # Function to update speaker name
    def update_speaker_column(example, index):
        new_speaker = f"TmpSpeaker_{index + 1}"
        return {'speaker': new_speaker}
    
    dataset = dataset.map(update_speaker_column, with_indices=True)

    if args.output_dir:
        print("Saving to disk...")
        dataset.save_to_disk(args.output_dir)
    
