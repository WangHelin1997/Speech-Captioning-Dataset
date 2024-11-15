import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import random
import numpy as np
import torch
from accelerate import Accelerator, skip_first_batches
from accelerate.logging import get_logger
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__, log_level="INFO")


@dataclass
class ModelArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    model_name_or_path: str = field(
        metadata={"help": "The name of the model to use (via the transformers library) for the prompt annotation."},
    )
    per_device_eval_batch_size: int = field(
        metadata={"help": "The per-device batch size to use for inference."},
    )
    model_variant: str = field(
        default=None,
        metadata={"help": "If specified load weights from `variant` filename, *e.g.* pytorch_model.<variant>.bin. "},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "Floating-point format in which the model weights should be initialized"
                " and the computations run. Choose one of `[float32, float16, bfloat16]`."
            )
        },
    )
    attn_implementation: Optional[str] = field(
        default="sdpa",
        metadata={"help": "Which attn type to use: ['eager', 'sdpa', 'flash_attention_2']"},
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 8-bit precision for inference."}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use 4-bit precision for inference."}
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"}
    )
    use_bnb_nested_quant: Optional[bool] = field(default=False, metadata={"help": "use nested quantization"})
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer for encoding/decoding input ids"}
    )
    token: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether or not to use an authentication token when loading/uploading from the Hugging Face Hub"
        },
    )
    do_sample: Optional[bool] = field(default=True, metadata={"help": "Whether to use sampling mode for generation"})
    temperature: Optional[float] = field(default=0.6, metadata={"help": "Temperature for sampling-based generation"})
    max_new_tokens: Optional[int] = field(
        default=500, metadata={"help": "Maximum number of new tokens during generation"}
    )
    torch_compile: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to compile the forward pass (not sampling) in generate. Only compatible with Gemma and LlaMA."
        },
    )
    num_description: Optional[int] = field(
        default=1, metadata={"help": "the number of description generation"}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    output_dir: str = field(
        metadata={
            "help": "Where to save the processed dataset to disk. If unspecified, uses a 'pretty' version of the "
            "original dataset name. E.g. 'facebook/voxpopuli' will be saved under 'voxpopuli'."
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    dataset_split_name: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the dataset to use (via the datasets library)."},
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cache directory for saving and loading datasets"},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum number of samples for generation - use for debugging purposes."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    dataloader_num_workers: Optional[int] = field(
        default=0,
        metadata={"help": "The number of processes to use for the dataloader."},
    )
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
    )
    hub_dataset_id: Optional[str] = field(
        default=None,
        metadata={"help": "Repository namespace if pushing to the Hugging Face Hub."},
    )
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory each time the script is run."},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the generated prompts every save_steps."},
    )
    save_total_limit: Optional[int] = field(
        default=1, metadata={"help": ("If a value is passed, will limit the total number of saved checkpoints")}
    )

    def __post_init__(self):
        if self.push_to_hub and self.hub_dataset_id is None:
            raise ValueError("You must specify the `hub_dataset_id` when setting `--push_to_hub=True`")


def get_quantization_config(model_args: ModelArguments) -> Union[BitsAndBytesConfig, None]:
    if model_args.load_in_4bit:
        compute_dtype = torch.float16
        if model_args.torch_dtype not in {"auto", None}:
            compute_dtype = getattr(torch, model_args.torch_dtype)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def get_current_device() -> int:
    """Get the current device. For GPU we return the local process index to enable multiple GPU training."""
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def get_kbit_device_map() -> Union[Dict[str, int], None]:
    """Useful for running inference with quantized models by setting `device_map=get_peft_device_map()`"""
    return {"": get_current_device()} if torch.cuda.is_available() else None


CHECKPOINT_PREFIX = "checkpoint"
_RE_CHECKPOINT = re.compile(r"^checkpoint-(\d+).json$")


def save_checkpoint(output_dir, all_generated_ids, step):
    checkpoint_path = f"{CHECKPOINT_PREFIX}-{step}.json"
    output_path = os.path.join(output_dir, checkpoint_path)
    all_generated_ids = [ids.tolist() for ids in all_generated_ids]
    with open(output_path, "w") as file:
        json.dump(all_generated_ids, file)


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path, "r") as file:
        all_generated_ids = json.load(file)
    all_generated_ids = [np.array(lst) for lst in all_generated_ids]
    return all_generated_ids


def sorted_checkpoints(output_dir=None) -> List[str]:
    """Helper function to sort saved checkpoints from oldest to newest."""
    ordering_and_checkpoint_path = []

    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{CHECKPOINT_PREFIX}-*")]

    for path in glob_checkpoints:
        regex_match = re.match(f".*{CHECKPOINT_PREFIX}-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def rotate_checkpoints(save_total_limit=None, output_dir=None) -> None:
    """Helper function to delete old checkpoints."""
    if save_total_limit is None or save_total_limit <= 0:
        return
    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = sorted_checkpoints(output_dir=output_dir)
    if len(checkpoints_sorted) <= save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
        os.remove(checkpoint)


def get_last_checkpoint(folder) -> Tuple[List, int]:
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
        return [], 0
    content = os.listdir(folder)
    checkpoints = [path for path in content if _RE_CHECKPOINT.search(path) is not None]
    if len(checkpoints) == 0:
        return [], 0
    last_checkpoint = os.path.join(folder, max(checkpoints, key=lambda x: int(_RE_CHECKPOINT.search(x).groups()[0])))
    # Find num steps saved state string pattern
    pattern = r"checkpoint-(\d+).json"
    match = re.search(pattern, last_checkpoint)
    cur_step = int(match.group(1))
    # load corresponding generated ids
    all_generated_ids = load_checkpoint(last_checkpoint)
    return all_generated_ids, cur_step


@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received to the longest sequence in the batch.
    """

    tokenizer: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_ids = {"input_ids": [feature["input_ids"] for feature in features]}
        batch = self.tokenizer.pad(input_ids, return_tensors="pt", padding="longest", return_attention_mask=True)
        return batch

PROMPT_FRONT = """
Objective:
Generate a single text description of a speech sample using the provided keywords.

Keywords:
1. Gender (e.g., male, female)
2. Age (e.g., teenager, adult, senior)
3. Brightness of the timbre (e.g., bright, dark)
4. Smoothness of the timbre (e.g., smooth, rough)
5. Emotion (e.g., angry, sad, happy, etc.)
6. Reverberation (e.g., very roomy sounding, quite confined sounding, etc.)
7. Noise level (e.g., very noisy, quite clear, etc.)
8. Tone (e.g., very monotone, quite expressive, etc.)
9. Pace (e.g., very slowly, quite fast, etc.)
10. Pitch (e.g., very low pitch, quite high pitch, etc.)

Instructions:
1. Use the keywords to create a grammatically correct and easy-to-understand description of the speech sample, varying the sentence structure and phrasing as much as possible across examples.
2. Rearrange the keyword order, split ideas across multiple sentences, or introduce descriptive transitions to make the description fluid and natural.
3. Substitute synonymous terms where appropriate, and rephrase parts of the description to add variety and keep it engaging.
4. If a keyword is 'None,' omit it from the description.
5. If noise is 'very noisy' and reverberation is 'very roomy sounding,' describe it as a 'very bad recording.'
6. If noise is 'very clear' and reverberation is 'very confined sounding,' describe it as a 'very good recording.'
7. Avoid repeating the same structure for consecutive descriptions. Explore different ways to convey the tone, pace, emotion, and other characteristics.
8. You can drop some of the keywords for diversity.
9. Return only the generated description.

"""

PROMPT_END = """
Keywords:
'[gender]', '[age]', '[brightness]', '[smoothness]', '[emotion]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]'
The corresponding description is:

"""

EXAMPLES = [
    """Given the keywords:
         'female', 'teenager', 'slightly bright', 'smooth', 'neutral', 'slightly confined sounding', 'very clear', 'somewhat expressive', 'medium pitch', 'average pace',
    A valid description could be: "A teenage girl speaks with a neutral tone, her smooth, slightly bright voice carrying a medium pitch. The sound is clear and confined, her words somewhat expressive."
    """,
    """Given the keywords:
         'male', 'teenager', 'dark', 'rough', 'angry', 'slightly confined sounding', 'moderately noisy', 'somewhat monotone', 'low pitch', 'quite fast',
    A valid description could be: "A teenage boy speaks with a low, dark voice that carries a rough, angry tone. His words come fast and slightly monotone, with a bit of background noise adding to the confined atmosphere."
    """,
    """Given the keywords:
         'female', 'adult', 'bright', 'smooth', 'happy', 'quite roomy sounding', 'very clear', 'expressive', 'high pitch', 'moderate pace',
    A valid description could be: "An adult woman speaks happily with a bright, smooth timbre and high pitch. Her voice resonates expressively through the clear, spacious room, flowing at a moderate pace."
    """,
    """Given the keywords:
         'male', 'senior', 'soft', 'smooth', 'neutral', 'quite confined sounding', 'very clear', 'somewhat monotone', 'medium pitch', 'slowly',
    A valid description could be: "Soft and smooth, an older man's neutral voice echoes in a very clear, confined space. His words are slow and somewhat monotone, with a steady, medium pitch."
    """,
    """Given the keywords:
         'female', 'child', 'slightly bright', 'rough', 'excited', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'quite high pitch', 'quite fast',
    A valid description could be: "A child’s voice, high-pitched and filled with excitement, is bright and slightly rough. She speaks expressively and fast, her tone echoing lightly with a touch of background noise."
    """,
    """Given the keywords:
         'male', 'adult', 'dark', 'smooth', 'calm', 'very confined sounding', 'very clear', 'monotone', 'low pitch', 'slow pace',
    A valid description could be: "A calm, low-pitched voice from an adult male drifts through a confined, very clear setting. The tone is dark and smooth, with a slow, monotone delivery."
    """,
    """Given the keywords:
         'female', 'teenager', 'bright', 'slightly rough', 'sad', 'roomy sounding', 'moderately noisy', 'somewhat expressive', 'low pitch', 'slow pace',
    A valid description could be: "With a touch of sadness, a teenage girl’s voice—bright yet slightly rough—fills a roomy space. Her low pitch is delivered slowly, and there’s a moderate level of background noise."
    """,
    """Given the keywords:
         'male', 'adult', 'slightly dark', 'smooth', 'neutral', 'slightly confined sounding', 'very clear', 'monotone', 'medium pitch', 'average pace',
    A valid description could be: "An adult male’s voice, neutral and slightly dark, sounds clear and confined. He speaks at a medium pitch and an even pace, with a steady, monotone rhythm."
    """,
    """Given the keywords:
         'female', 'senior', 'bright', 'smooth', 'happy', 'quite roomy sounding', 'clear', 'expressive', 'high pitch', 'quite fast',
    A valid description could be: "Her high-pitched, bright voice fills the room with clear, expressive energy. The woman speaks quickly, a note of happiness in her tone."
    """,
    """Given the keywords:
         'male', 'child', 'soft', 'slightly rough', 'curious', 'slightly roomy sounding', 'slightly noisy', 'expressive', 'quite high pitch', 'slowly',
    A valid description could be: "A curious young boy speaks slowly in a soft, slightly rough tone. His voice is high-pitched, expressive, and gently echoes through the room with light background noise."
    """,
    """Given the keywords:
         'female', 'adult', 'bright', 'smooth', 'neutral', 'quite confined sounding', 'very clear', 'monotone', 'medium pitch', 'quite fast',
    A valid description could be: "In a confined, very clear space, an adult woman’s bright, smooth voice comes across in a neutral, monotone manner. She speaks at a quick pace with a moderate pitch."
    """,
    """Given the keywords:
         'male', 'teenager', 'slightly dark', 'rough', 'angry', 'roomy sounding', 'noisy', 'very expressive', 'low pitch', 'fast',
    A valid description could be: "A teenage boy’s voice, low-pitched and dark, is rough with anger. His words spill out quickly and expressively, filling a noisy, open room."
    """,
    """Given the keywords:
         'female', 'child', 'bright', 'smooth', 'cheerful', 'quite confined sounding', 'quite clear', 'expressive', 'high pitch', 'moderate pace',
    A valid description could be: "In a cheerful tone, a young girl speaks with a bright, smooth voice. Her high pitch and expressive delivery ring out clearly in a confined space."
    """,
    """Given the keywords:
         'male', 'adult', 'dark', 'rough', 'bored', 'slightly roomy sounding', 'slightly noisy', 'monotone', 'low pitch', 'slowly',
    A valid description could be: "A man’s voice, dark and rough, comes slowly and monotonously, conveying a sense of boredom. There’s a slight echo and faint noise in the background."
    """,
    """Given the keywords:
         'female', 'senior', 'bright', 'smooth', 'relaxed', 'quite confined sounding', 'very clear', 'somewhat expressive', 'medium pitch', 'average pace',
    A valid description could be: "A relaxed, smooth voice with a bright timbre belongs to an older woman. Her medium pitch sounds very clear and slightly expressive within a confined space."
    """,
    """Given the keywords:
         'male', 'teenager', 'slightly dark', 'smooth', 'curious', 'roomy sounding', 'clear', 'expressive', 'medium pitch', 'quite fast',
    A valid description could be: "A curious teenage boy speaks at a quick pace in a slightly dark, smooth tone. His medium pitch echoes expressively through a clear, spacious room."
    """,
    """Given the keywords:
         'female', 'child', 'soft', 'gentle', 'calm', 'slightly roomy sounding', 'quite clear', 'monotone', 'high pitch', 'slow pace',
    A valid description could be: "Calm and gentle, a young girl’s soft, high-pitched voice fills a slightly roomy, clear space. She speaks at a slow pace, her tone largely monotone."
    """,
    """Given the keywords:
         'male', 'adult', 'dark', 'rough', 'neutral', 'confined sounding', 'very clear', 'monotone', 'low pitch', 'quite fast',
    A valid description could be: "An adult man’s voice, dark and rough, delivers words quickly in a monotone style. The recording is very clear, and his low pitch reverberates within a confined space."
    """,
    """Given the keywords:
         'female', 'adult', 'bright', 'smooth', 'excited', 'slightly roomy sounding', 'slightly noisy', 'expressive', 'high pitch', 'very fast',
    A valid description could be: "Excitement fills an adult woman’s bright, smooth voice as she speaks expressively at a very fast pace. Her high pitch carries in a slightly echoey room with mild noise."
    """,
    """Given the keywords:
         'male', 'senior', 'soft', 'slightly rough', 'sad', 'roomy sounding', 'moderate noise level', 'somewhat expressive', 'low pitch', 'slowly',
    A valid description could be: "A low, sad voice with a soft, slightly rough edge emerges from an elderly man. His words come slowly, echoing in a moderately noisy room."
    """
]


def main():
    # 1. Parse input arguments
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # 2. Setup logging
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    accelerator = Accelerator()

    if data_args.overwrite_output_dir and os.path.exists(data_args.output_dir) and os.path.isdir(data_args.output_dir):
        logger.info("Cleaning output dir from previous run...")
        shutil.rmtree(data_args.output_dir)

    # 3. Load annotated dataset
    logger.info("*** Load annotated dataset ***")
    if data_args.dataset_split_name is not None:
        raw_datasets = DatasetDict()
        data_splits = data_args.dataset_split_name.split("+")
        # load on a split-wise basis
        for split in data_splits:
            with accelerator.local_main_process_first():
                raw_datasets[split] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=split,
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    num_proc=data_args.preprocessing_num_workers,
                )
    else:
        with accelerator.local_main_process_first():
            # load all splits for annotation
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                num_proc=data_args.preprocessing_num_workers,
            )

    raw_datasets_features = set(raw_datasets[next(iter(raw_datasets))].features.keys())

    if data_args.max_eval_samples is not None:
        for split in raw_datasets:
            raw_datasets[split] = raw_datasets[split].select(range(data_args.max_eval_samples))

    # TODO(SG): add accent
    EXPECTED_COLUMNS = {"pitch", "noise", "reverberation", "speech_monotony", "speaking_rate", 'age','accent','brightness','emotion','gender','smoothness'}
    if not EXPECTED_COLUMNS.issubset(raw_datasets_features):
        missing_columns = EXPECTED_COLUMNS - raw_datasets_features
        raise ValueError(
            f"Missing columns {missing_columns} from the dataset features. Got dataset features {raw_datasets_features}"
        )

    # 4. Load pre-trained model
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        variant=model_args.model_variant,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        token=model_args.token,
        cache_dir=model_args.cache_dir
    ).eval()

    if model_args.torch_compile:
        # torch compile only compatible with gemma and llama
        if not callable(getattr(model, "_setup_cache", None)):
            raise ValueError(
                f"Static k/v cache is not compatible with the model {model.__class__.__name__}. Set `--torch_compile=False"
                "for dynamic k/v cache"
            )
        model.generation_config.cache_implementation = "static"
        # compile the forward pass (but not the top-{p,k} sampling)
        model = torch.compile(model, mode="reduce-overhead", fullgraph=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

    def prepare_dataset(sample):
        examples = random.sample(EXAMPLES, 3)
        sample_prompt = PROMPT_FRONT + "Examples:\n1. " + examples[0] + "2. " + examples[1] + "3. " + examples[2] + PROMPT_END
        for key in EXPECTED_COLUMNS:
            sample_prompt = sample_prompt.replace(f"[{key}]", sample[key])
        sample_prompt = [{"role": "user", "content": sample_prompt}]
        token_ids = tokenizer.apply_chat_template(sample_prompt)
        sample["input_ids"] = token_ids
        return sample

    with accelerator.local_main_process_first():
        vectorized_datasets = raw_datasets.map(
            prepare_dataset, num_proc=data_args.preprocessing_num_workers, desc="Preparing prompts"
        )

    # Prepare everything with our `accelerator`
    model = accelerator.prepare(model)
    data_collator = DataCollatorWithPadding(tokenizer)

    def generate_step(batch):
        output_ids = accelerator.unwrap_model(model).generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            do_sample=model_args.do_sample,
            temperature=model_args.temperature,
            max_new_tokens=model_args.max_new_tokens,
        )
        output_ids = accelerator.pad_across_processes(output_ids, dim=1, pad_index=tokenizer.pad_token_id)
        return output_ids

    def postprocess_dataset(sample):
        prompt_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        generated_text = tokenizer.decode(sample["generated_ids"], skip_special_tokens=True)

        sample["text_description"+str(model_args.num_description)] = generated_text[len(prompt_text) :]
        return sample

    for split in vectorized_datasets:
        data_loader = DataLoader(
            vectorized_datasets[split],
            batch_size=model_args.per_device_eval_batch_size,
            collate_fn=data_collator,
            num_workers=data_args.dataloader_num_workers,
            pin_memory=True,
        )
        data_loader = accelerator.prepare(data_loader)
        total_inference_steps = len(data_loader)
        progress_bar = tqdm(
            range(total_inference_steps), desc=" ... ", position=0, disable=not accelerator.is_local_main_process
        )

        split_output_dir = os.path.join(data_args.output_dir, split)
        all_generated_ids, cur_step = get_last_checkpoint(split_output_dir)

        if cur_step > 0:
            logger.info(f"Resuming {split} from step {cur_step}")
            # efficiently skip the first n batches
            data_loader = skip_first_batches(data_loader, cur_step)
            progress_bar.update(cur_step)

        while cur_step < total_inference_steps:
            for batch in data_loader:
                generated_ids = generate_step(batch)
                generated_ids = accelerator.gather_for_metrics(generated_ids)
                all_generated_ids.extend(generated_ids.cpu().numpy())

                cur_step += 1
                progress_bar.update(1)

                if (cur_step % data_args.save_steps == 0) or (cur_step == total_inference_steps):
                    save_checkpoint(split_output_dir, all_generated_ids, cur_step)
                    rotate_checkpoints(data_args.save_total_limit, output_dir=split_output_dir)

        vectorized_datasets[split] = vectorized_datasets[split].add_column("generated_ids", all_generated_ids)

        if accelerator.is_main_process:
            vectorized_datasets[split] = vectorized_datasets[split].map(
                postprocess_dataset,
                num_proc=data_args.preprocessing_num_workers,
                desc="Postprocessing dataset",
                remove_columns=["input_ids", "generated_ids"],
            )
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        if data_args.push_to_hub:
            vectorized_datasets.push_to_hub(
                data_args.hub_dataset_id,
                config_name=data_args.dataset_config_name if data_args.dataset_config_name is not None else "default",
                token=model_args.token,
            )
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
