from datasets import load_dataset, Audio, load_from_disk
from multiprocess import set_start_method
from dataspeech import rate_apply, pitch_apply, snr_apply, squim_apply
import torch
import argparse
import os
import re

os.environ['OPENBLAS_NUM_THREADS'] = '1'


if __name__ == "__main__":
    set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", default=None, type=str, help="Cache dir to download data")
    parser.add_argument("--output_dir", default=None, type=str, help="If specified, save the dataset on disk with this path.")
    parser.add_argument("--repo_id", default=None, type=str, help="If specified, push the dataset to the hub.")
    parser.add_argument("--audio_column_name", default="audio", type=str, help="Column name of the audio column to be enriched.")
    parser.add_argument("--text_column_name", default="text", type=str, help="Text column name.")
    parser.add_argument("--rename_column", action="store_true", help="If activated, rename audio and text column names to 'audio' and 'text'. Useful if you want to merge datasets afterwards.")
    parser.add_argument("--cpu_num_workers", default=1, type=int, help="Number of CPU workers for transformations that don't use GPUs or if no GPU are available.")
    parser.add_argument("--cpu_writer_batch_size", default=1000, type=int, help="writer_batch_size for transformations that don't use GPUs. See: https://huggingface.co/docs/datasets/v2.17.0/en/package_reference/main_classes#datasets.Dataset.map.writer_batch_size")
    parser.add_argument("--batch_size", default=2, type=int, help="This parameters specify how many samples are passed by workers for operations that are using GPUs.")
    parser.add_argument("--penn_batch_size", default=4096, type=int, help="Pitch estimation chunks audio into smaller pieces and processes them in batch. This specify the batch size. If you are using a gpu, pick a batch size that doesn't cause memory errors.")
    parser.add_argument("--num_workers_per_gpu_for_pitch", default=1, type=int, help="Number of workers per GPU for the pitch estimation if GPUs are available. Defaults to 1 if some are avaiable. Useful if you want multiple processes per GPUs to maximise GPU usage.")
    parser.add_argument("--num_workers_per_gpu_for_snr", default=1, type=int, help="Number of workers per GPU for the SNR and reverberation estimation if GPUs are available. Defaults to 1 if some are avaiable. Useful if you want multiple processes per GPUs to maximise GPU usage.")
    parser.add_argument("--apply_squim_quality_estimation", action="store_true", help="If set, will also use torchaudio-squim estimation (SI-SNR, STOI and PESQ).")
    parser.add_argument("--num_workers_per_gpu_for_squim", default=1, type=int, help="Number of workers per GPU for the SI-SNR, STOI and PESQ estimation if GPUs are available. Defaults to 1 if some are avaiable. Useful if you want multiple processes per GPUs to maximise GPU usage.")


    args = parser.parse_args()
    print(torch.cuda.device_count())
    dataset = load_from_disk(args.cache_dir)

    if args.apply_squim_quality_estimation:
        print("Compute SI-SDR, PESQ, STOI")
        squim_dataset = dataset.map(
            squim_apply,
            batched=True,
            batch_size=args.batch_size,
            with_rank=True if torch.cuda.device_count()>0 else False,
            num_proc=torch.cuda.device_count()*args.num_workers_per_gpu_for_squim if torch.cuda.device_count()>0 else args.cpu_num_workers,
            remove_columns=["audio"], # tricks to avoid rewritting audio
            fn_kwargs={"audio_column_name": "audio",},
        )

    print("Compute pitch")
    pitch_dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000)).map(
        pitch_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count()>0 else False,
        num_proc=torch.cuda.device_count()*args.num_workers_per_gpu_for_pitch if torch.cuda.device_count()>0 else args.cpu_num_workers,
        # remove_columns=[audio_column_name], # tricks to avoid rewritting audio
        fn_kwargs={"audio_column_name": "audio", "penn_batch_size": args.penn_batch_size},
    )

    print("Compute snr and reverb")
    snr_dataset = dataset.map(
        snr_apply,
        batched=True,
        batch_size=args.batch_size,
        with_rank=True if torch.cuda.device_count()>0 else False,
        num_proc=torch.cuda.device_count()*args.num_workers_per_gpu_for_snr if torch.cuda.device_count()>0 else args.cpu_num_workers,
        # remove_columns=["audio"], # tricks to avoid rewritting audio
        fn_kwargs={"audio_column_name": "audio"},
    )
    
    print("Compute speaking rate")
    if "speech_duration" in snr_dataset[0].keys():    
        rate_dataset = snr_dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size= args.cpu_writer_batch_size,
            fn_kwargs={"audio_column_name": "audio", "text_column_name": "text"},
        )
    else:
        rate_dataset = dataset.map(
            rate_apply,
            with_rank=False,
            num_proc=args.cpu_num_workers,
            writer_batch_size= args.cpu_writer_batch_size,
            remove_columns=["audio"], # tricks to avoid rewritting audio
            fn_kwargs={"audio_column_name": "audio", "text_column_name": "text"},
        )
    
    dataset = pitch_dataset.add_column("snr", snr_dataset["snr"]).add_column("c50", snr_dataset["c50"])
    if "speech_duration" in snr_dataset:
        dataset = dataset.add_column("speech_duration", snr_dataset["snr"])
    dataset = dataset.add_column("speaking_rate", rate_dataset["speaking_rate"]).add_column("phonemes", rate_dataset["phonemes"])
    if args.apply_squim_quality_estimation:
        dataset = dataset.add_column("stoi", squim_dataset["stoi"]).add_column("si-sdr", squim_dataset["sdr"]).add_column("pesq", squim_dataset["pesq"])

    print("Saving to disk...")
    dataset.save_to_disk(args.output_dir)
