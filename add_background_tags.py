from datasets import load_from_disk
import argparse
from huggingface_hub import create_repo, login
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val_cleaned/", type=str)
    parser.add_argument("--tagpath", default="/data/lmorove1/hwang258/Speech-Captioning-Dataset/audio_tagging.txt", type=str)
    parser.add_argument("--savepath", default="/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val_cleaned_eat/", type=str)
    parser.add_argument("--repo", default="OpenSound/GigaCaps_val_3", type=str)
    args = parser.parse_args()
    
    os.makedirs(args.savepath.rsplit('/',1)[0], exist_ok=True)

    datadir = args.datadir
    savepath = args.savepath
    dataset = load_from_disk(datadir)
    
    with open(args.tagpath, "r") as f:
        ids = [line.strip() for line in f.readlines()]
    dicts = {}
    for id in ids:
        if len(id.split('\t')) == 1:
            dicts[id.split('\t')[0]] = 'None'
        else:
            dicts[id.split('\t')[0]] = id.split('\t')[1]
        
    print(len(dicts))

    def add_column(example):
        id = example["segment_id"]
        example["background"] = dicts[id] if id in dicts.keys() else "None"
        return example

    dataset = dataset.map(add_column)
    dataset.save_to_disk(args.savepath)
    
    # create_repo(repo_id=args.repo, repo_type="dataset", private=True)
    # dataset.push_to_hub(repo_id=args.repo, max_shard_size="100MB", private=True)

