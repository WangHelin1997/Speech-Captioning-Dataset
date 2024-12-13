from datasets import load_from_disk
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", default="/data/lmorove1/hwang258/gigadata/GigaCaps3/gigaspeech_val_cleaned/", type=str)
    parser.add_argument("--savepath", default="/data/lmorove1/hwang258/Speech-Captioning-Dataset/noisy_ids.txt", type=str)
    args = parser.parse_args()

    datadir = args.datadir
    savepath = args.savepath
    dataset = load_from_disk(datadir)

    def get_ids_noisy(example, idx):
        noise = example['noise']
        is_noisy = (noise=='slightly noisy' or noise=='quite noisy' or noise=='very noisy')
        return {"segment_id": example['segment_id'], "is_noisy": is_noisy}

    ids_noisy = dataset.map(
        get_ids_noisy,
        with_indices=True,  # Provide row indices
        remove_columns=dataset.column_names  # Remove other columns for simplicity
    )
    ids_noisy = [row["segment_id"] for row in ids_noisy if row["is_noisy"]]
    with open(savepath, "w") as f:
        for item in ids_noisy:
            f.write(f"{item}\n")
