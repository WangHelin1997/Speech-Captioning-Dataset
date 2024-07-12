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


# TODO(SG): add accent keyword
# PROMPT = """You will be given eleven descriptive keywords related to an audio sample of a person's speech. These keywords include:
# 1. The gender (e.g., male, female)
# 2. The age (e.g., teenagers, adult, senior)
# 3. The brightness of the timbre (e.g. bright, dark)
# 4. The smoothness of the timbre (e.g. smooth, rough)
# 5. The accent (e.g. Dutch, German, Czech, Polish, French, Hungarian, Finnish, Romanian, Slovak, Spanish, Italian, Estonian, Lithuanian, Croatian, Slovene, English, Scottish, Irish, NorthernIrish, Indian, Vietnamese, Canadian, American)
# 6. The emotion (e.g. angry, disgust, sad, fear, happy, neutral)
# 7. The level of reverberation (e.g., very roomy sounding, quite roomy sounding, slightly roomy sounding, moderate reverberation, slightly confined sounding, quite confined sounding, very confined sounding)
# 8. The amount of noise the sample (e.g., very noisy, quite noisy, slightly noisy, moderate ambient sound, slightly clear, quite clear, very clear)
# 9. The tone of the speaker's voice (e.g., very monotone, quite monotone, slightly monotone, moderate intonation, slightly expressive, quite expressive, very expressive)
# 10. The pace of the speaker's delivery (e.g., very slowly, quite slowly, slightly slowly, moderate speed, slightly fast, quite fast, very fast)
# 11. The pitch of the speaker's voice (e.g., very low pitch, quite low pitch, slightly low pitch, moderate pitch, slightly high pitch, quite high pitch, very high pitch)

# Your task is to create a text description using these keywords that accurately describes the speech sample while ensuring the description remains grammatically correct and easy to understand. You should rearrange the keyword order as necessary, and substitute synonymous terms where appropriate. If the term is None, just remove that keyword for the speech sample. If the amount of noise is 'very noisy' and the level of reverberation is 'very roomy sounding', include terms like 'very bad recording' in the description. Likewise, if the amount of noise is 'very clear' and the level of reverberation is 'very confined sounding', include terms like 'very good recording' in the description. Otherwise, do not add extra details beyond what has been provided, and only return the generated description.

# For example, given the following keywords: 'female', 'adult', 'bright', 'smooth', 'None', 'happy', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly', a valid description would be: 'an adult woman with a deep, bright and smooth voice speaks slowly and happily but has an animated delivery in an echoey room with some background noise'.
# Another valid description would be: 'In a room with slight background noise, a female middle-aged speaker delivers an animated, bright, smooth and expressive speech, at a very slow pace, with a happy mood.'

# For the keywords: '[gender]', '[age]', '[brightness]', '[smoothness]', '[accent]', '[emotion]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]', the corresponding description is:"
# """
PROMPT_FRONT = """
Objective:
Generate a single text description of a speech sample using the provided keywords.

Keywords:
1. Gender (e.g., male, female)
2. Age (e.g., teenager, adult, senior)
3. Brightness of the timbre (e.g., bright, dark)
4. Smoothness of the timbre (e.g., smooth, rough)
5. Accent (e.g., Dutch, German, etc.)
6. Emotion (e.g., angry, sad, happy, etc.)
7. Reverberation (e.g., very roomy sounding, quite confined sounding, etc.)
8. Noise level (e.g., very noisy, quite clear, etc.)
9. Tone (e.g., very monotone, quite expressive, etc.)
10. Pace (e.g., very slowly, quite fast, etc.)
11. Pitch (e.g., very low pitch, quite high pitch, etc.)

Instructions:
1. Use these keywords to create a grammatically correct and easy-to-understand description of the speech sample.
2. Rearrange the keyword order as necessary and substitute synonymous terms where appropriate.
3. If a keyword is 'None,' omit it from the description.
4. If noise is 'very noisy' and reverberation is 'very roomy sounding,' mention 'very bad recording.'
5. If noise is 'very clear' and reverberation is 'very confined sounding,' mention 'very good recording.'
6. Do not add extra details beyond the provided keywords.
7. You can drop one of two keywords for diversity.
8. Return only the generated description.


"""

PROMPT_END = """
Keywords:
'[gender]', '[age]', '[brightness]', '[smoothness]', '[accent]', '[emotion]', '[reverberation]', '[noise]', '[speech_monotony]', '[pitch]', '[speaking_rate]'
The corresponding description is:

"""

EXAMPLES = [
"""Given the keywords:
'female', 'adult', 'bright', 'smooth', 'American', 'happy', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'slightly low pitch', 'very slowly'
A valid description could be: "An adult American woman with a bright, smooth voice speaks very slowly and happily with animated delivery in an echoey room with some background noise."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'Scottish', 'angry', 'very confined sounding', 'very clear', 'very monotone', 'very low pitch', 'moderate speed'
A valid description could be: "An elderly Scottish man with a dark, rough voice speaks at a moderate speed, sounding angry and monotone in a very clear, confined room, making it a very good recording."
""",
"""Given the keywords: 'female', 'teenager', 'bright', 'smooth', 'French', 'neutral', 'moderate reverberation', 'moderate ambient sound', 'slightly expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenage girl with a bright, smooth voice speaks quickly and neutrally with slight expressiveness in a moderately reverberant room with ambient noise."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'German', 'happy', 'very roomy sounding', 'very noisy', 'very expressive', 'very high pitch', 'very fast'
A valid description could be: "A joyful German man with a high-pitched, dark, smooth voice speaks very quickly and expressively. However, the very noisy, roomy environment makes it a very bad recording."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'moderate pitch', 'slightly slowly'
A valid description could be: "A senior Canadian woman delivers her speech slowly with a bright yet rough voice, clearly heard in a slightly confined space. Her sadness is palpable despite the clarity of the recording."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'rough', 'None', 'angry', 'very confined sounding', 'very clear', 'very monotone', 'very low pitch', 'very slowly'
A valid description could be: "A teenage boy speaks angrily with a dark, rough voice, his words coming out slowly and monotonously. The recording is very clear and confined."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'None', 'happy', 'slightly roomy sounding', 'slightly noisy', 'very expressive', 'None', 'very slowly'
A valid description could be: "An adult woman with a bright, smooth voice speaks very slowly and happily with animated delivery in an echoey room with some background noise."
""",
"""Given the keywords: 
'male', 'senior', 'None', 'smooth', 'Irish', 'sad', 'moderate reverberation', 'moderate ambient sound', 'quite monotone', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly Irish man delivers a sad, monotone speech at a moderate speed. His smooth voice resonates moderately in the room filled with ambient sound."
""",
"""Given the keywords: 
'female', 'teenager', 'None', 'None', 'French', 'neutral', 'slightly roomy sounding', 'moderate ambient sound', 'slightly expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenager speaks quickly with a neutral tone, slightly expressive. The speech occurs in a moderately ambient, roomy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'None', 'German', 'happy', 'very roomy sounding', 'very noisy', 'very expressive', 'very high pitch', 'very fast'
A valid description could be: "An adult German man with a high-pitched, dark voice speaks very fast and happily with expressive delivery in a very noisy, echoey room, making it a very bad recording."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'None', 'Canadian', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'None', 'slightly slowly'
A valid description could be: "A senior Canadian woman speaks slowly with a bright voice, her sadness clear in her expressive tone. The recording is quite clear in a slightly confined space."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'rough', 'Scottish', 'None', 'moderate reverberation', 'slightly noisy', 'slightly monotone', 'very low pitch', 'moderate speed'
A valid description could be: "An adult Scottish man speaks at a moderate speed with a dark, rough voice. His speech is slightly monotone and the room has moderate reverberation and slight noise."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'None', 'happy', 'quite confined sounding', 'quite clear', 'slightly expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A happy teenage girl with a bright, smooth voice speaks quickly. The recording is quite clear and confined, with a slight expressiveness in her moderate-pitched voice."
""",
"""Given the keywords: 
'male', 'senior', 'None', 'smooth', 'Dutch', 'sad', 'very roomy sounding', 'very noisy', 'very monotone', 'slightly low pitch', 'very slowly'
A valid description could be: "An elderly Dutch man speaks very slowly with a smooth, low-pitched voice, sounding sad and monotone. The very noisy, roomy environment makes it a poor recording."
""",
"""Given the keywords: 
'female', 'adult', 'None', 'rough', 'Estonian', 'neutral', 'slightly confined sounding', 'moderate ambient sound', 'quite expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An Estonian woman in her prime speaks at a moderate speed with a rough, slightly high-pitched voice. Her neutral tone is quite expressive in the moderately ambient, slightly confined space."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'rough', 'American', 'angry', 'slightly confined sounding', 'quite clear', 'very expressive', 'slightly low pitch', 'quite fast'
A valid description could be: "A teenage American boy speaks quickly with a dark, rough voice, sounding angry and expressive. The recording is quite clear and slightly confined."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'smooth', 'French', 'happy', 'moderate reverberation', 'slightly noisy', 'quite expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly French woman speaks at a moderate speed with a bright, smooth voice, her speech happy and expressive despite the slight noise in the reverberant room."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Irish', 'sad', 'very confined sounding', 'quite clear', 'very monotone', 'very low pitch', 'very slowly'
A valid description could be: "A senior Irish man speaks very slowly with a dark, smooth voice, sounding sad and monotone in a very clear, confined room."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'rough', 'German', 'happy', 'quite roomy sounding', 'moderate ambient sound', 'slightly expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A German teenage girl speaks quickly with a bright, rough voice, her speech happy and slightly expressive in a moderately ambient, roomy environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'smooth', 'American', 'neutral', 'slightly confined sounding', 'slightly noisy', 'quite monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An elderly American man delivers a neutral, dark, and smooth speech at a moderate speed. The slightly noisy, confined environment adds to the monotone delivery."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Canadian', 'happy', 'slightly roomy sounding', 'quite clear', 'very expressive', 'slightly high pitch', 'very slowly'
A valid description could be: "An adult Canadian woman with a bright, smooth voice speaks very slowly and happily with expressive delivery in a slightly roomy, clear environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'rough', 'Dutch', 'angry', 'moderate reverberation', 'slightly noisy', 'slightly monotone', 'moderate pitch', 'quite fast'
A valid description could be: "A teenage Dutch boy speaks quickly with a dark, rough voice, sounding angry and slightly monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'smooth', 'American', 'neutral', 'very confined sounding', 'very clear', 'quite expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly American woman speaks at a moderate speed with a bright, smooth voice. Her neutral but expressive tone is clear in the confined environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'rough', 'French', 'sad', 'slightly roomy sounding', 'moderate ambient sound', 'very monotone', 'very low pitch', 'very slowly'
A valid description could be: "An adult French man speaks very slowly with a dark, rough voice, sounding sad and very monotone in a slightly roomy, moderately ambient room."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'Scottish', 'happy', 'moderate reverberation', 'quite clear', 'slightly expressive', 'slightly high pitch', 'quite fast'
A valid description could be: "A Scottish teenage girl speaks quickly with a bright, smooth voice, her speech happy and slightly expressive in a moderately reverberant, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'smooth', 'Irish', 'angry', 'slightly confined sounding', 'slightly noisy', 'very expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly Irish man speaks at a moderate speed with a dark, smooth voice, sounding angry and very expressive in a slightly noisy, confined room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'rough', 'Estonian', 'neutral', 'very roomy sounding', 'very noisy', 'slightly monotone', 'slightly low pitch', 'very slowly'
A valid description could be: "An adult Estonian woman speaks very slowly with a bright, rough voice, sounding neutral and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Canadian', 'happy', 'moderate reverberation', 'slightly noisy', 'very expressive', 'slightly high pitch', 'quite fast'
A valid description could be: "A Canadian teenage boy speaks quickly with a dark, smooth voice, sounding happy and very expressive in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'smooth', 'French', 'sad', 'slightly confined sounding', 'quite clear', 'quite monotone', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly French woman speaks at a moderate speed with a bright, smooth voice, sounding sad and quite monotone in a slightly confined, clear environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'rough', 'Scottish', 'angry', 'very roomy sounding', 'very noisy', 'very expressive', 'very low pitch', 'very fast'
A valid description could be: "An adult Scottish man speaks very fast with a dark, rough voice, sounding angry and very expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'Dutch', 'happy', 'slightly confined sounding', 'slightly noisy', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A Dutch teenage girl speaks quickly with a bright, smooth voice, sounding happy and quite expressive in a slightly confined, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'neutral', 'moderate reverberation', 'quite clear', 'slightly monotone', 'moderate pitch', 'very slowly'
A valid description could be: "An elderly American man speaks very slowly with a dark, rough voice, sounding neutral and slightly monotone in a moderately reverberant, clear environment."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'happy', 'very roomy sounding', 'very noisy', 'very expressive', 'very high pitch', 'quite fast'
A valid description could be: "An adult Irish woman speaks quickly with a bright, smooth voice, sounding happy and very expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'German', 'sad', 'slightly confined sounding', 'quite clear', 'very monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "A German teenage boy speaks at a moderate speed with a dark, smooth voice, sounding sad and very monotone in a slightly confined, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'neutral', 'moderate reverberation', 'slightly noisy', 'slightly expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "An elderly Canadian woman speaks at a moderate speed with a bright, rough voice, sounding neutral and slightly expressive in a moderately reverberant, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'American', 'angry', 'slightly roomy sounding', 'very noisy', 'quite monotone', 'very low pitch', 'very slowly'
A valid description could be: "An adult American man speaks very slowly with a dark, smooth voice, sounding angry and quite monotone in a very noisy, slightly roomy room."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'Dutch', 'happy', 'quite confined sounding', 'quite clear', 'very expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "A Dutch teenage girl speaks at a moderate speed with a bright, smooth voice, sounding happy and very expressive in a quite confined, clear room."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'Estonian', 'sad', 'moderate reverberation', 'slightly noisy', 'slightly monotone', 'moderate pitch', 'very slowly'
A valid description could be: "An elderly Estonian man speaks very slowly with a dark, rough voice, sounding sad and slightly monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'French', 'neutral', 'slightly confined sounding', 'quite clear', 'quite expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "An adult French woman speaks at a moderate speed with a bright, smooth voice, sounding neutral and quite expressive in a slightly confined, clear room."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Irish', 'angry', 'very roomy sounding', 'very noisy', 'very monotone', 'very low pitch', 'quite fast'
A valid description could be: "A teenage Irish boy speaks quickly with a dark, smooth voice, sounding angry and very monotone in a very noisy, roomy room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'German', 'happy', 'slightly roomy sounding', 'slightly noisy', 'quite expressive', 'moderate pitch', 'very slowly'
A valid description could be: "An elderly German woman speaks very slowly with a bright, rough voice, sounding happy and quite expressive in a slightly noisy, slightly roomy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Dutch', 'neutral', 'very confined sounding', 'quite clear', 'slightly monotone', 'moderate pitch', 'quite fast'
A valid description could be: "An adult Dutch man speaks quickly with a dark, smooth voice, sounding neutral and slightly monotone in a very clear, confined room."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'Canadian', 'sad', 'moderate reverberation', 'quite clear', 'very expressive', 'slightly high pitch', 'very slowly'
A valid description could be: "A Canadian teenage girl speaks very slowly with a bright, smooth voice, sounding sad and very expressive in a moderately reverberant, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'angry', 'slightly confined sounding', 'slightly noisy', 'quite monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An elderly American man speaks at a moderate speed with a dark, rough voice, sounding angry and quite monotone in a slightly confined, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'neutral', 'very roomy sounding', 'very noisy', 'slightly expressive', 'very high pitch', 'quite fast'
A valid description could be: "An adult Irish woman speaks quickly with a bright, smooth voice, sounding neutral and slightly expressive in a very noisy, roomy room."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'German', 'happy', 'moderate reverberation', 'quite clear', 'very monotone', 'moderate pitch', 'moderate speed'
A valid description could be: "A German teenage boy speaks at a moderate speed with a dark, smooth voice, sounding happy and very monotone in a moderately reverberant, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'sad', 'slightly confined sounding', 'very clear', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly Canadian woman speaks quickly with a bright, rough voice, sounding sad and quite expressive in a very clear, slightly confined room."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Estonian', 'angry', 'very roomy sounding', 'very noisy', 'slightly monotone', 'very low pitch', 'very slowly'
A valid description could be: "An adult Estonian man speaks very slowly with a dark, smooth voice, sounding angry and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'French', 'neutral', 'slightly confined sounding', 'quite clear', 'quite expressive', 'moderate pitch', 'moderate speed'
A valid description could be: "A French teenage girl speaks at a moderate speed with a bright, smooth voice, sounding neutral and quite expressive in a slightly confined, clear room."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'happy', 'moderate reverberation', 'slightly noisy', 'very monotone', 'slightly low pitch', 'quite fast'
A valid description could be: "An elderly American man speaks quickly with a dark, rough voice, sounding happy and very monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'sad', 'very roomy sounding', 'very noisy', 'slightly expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An adult Irish woman speaks at a moderate speed with a bright, smooth voice, sounding sad and slightly expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Dutch', 'angry', 'slightly confined sounding', 'quite clear', 'quite monotone', 'moderate pitch', 'very slowly'
A valid description could be: "A Dutch teenage boy speaks very slowly with a dark, smooth voice, sounding angry and quite monotone in a slightly confined, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'happy', 'moderate reverberation', 'slightly noisy', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly Canadian woman speaks quickly with a bright, rough voice, sounding happy and quite expressive in a moderately reverberant, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Estonian', 'neutral', 'very roomy sounding', 'very noisy', 'slightly monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An adult Estonian man speaks at a moderate speed with a dark, smooth voice, sounding neutral and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'French', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenage girl speaks quickly with a bright, smooth voice, sounding sad and very expressive in a slightly confined, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'angry', 'moderate reverberation', 'slightly noisy', 'quite monotone', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly American man speaks quickly with a dark, rough voice, sounding angry and quite monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'neutral', 'very roomy sounding', 'very noisy', 'slightly expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An adult Irish woman speaks at a moderate speed with a bright, smooth voice, sounding neutral and slightly expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Dutch', 'happy', 'slightly confined sounding', 'quite clear', 'very monotone', 'moderate pitch', 'very slowly'
A valid description could be: "A Dutch teenage boy speaks very slowly with a dark, smooth voice, sounding happy and very monotone in a slightly confined, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'sad', 'moderate reverberation', 'slightly noisy', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly Canadian woman speaks quickly with a bright, rough voice, sounding sad and quite expressive in a moderately reverberant, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Estonian', 'neutral', 'very roomy sounding', 'very noisy', 'slightly monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An adult Estonian man speaks at a moderate speed with a dark, smooth voice, sounding neutral and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'French', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenage girl speaks quickly with a bright, smooth voice, sounding sad and very expressive in a slightly confined, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'angry', 'moderate reverberation', 'slightly noisy', 'quite monotone', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly American man speaks quickly with a dark, rough voice, sounding angry and quite monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'neutral', 'very roomy sounding', 'very noisy', 'slightly expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An adult Irish woman speaks at a moderate speed with a bright, smooth voice, sounding neutral and slightly expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Dutch', 'happy', 'slightly confined sounding', 'quite clear', 'very monotone', 'moderate pitch', 'very slowly'
A valid description could be: "A Dutch teenage boy speaks very slowly with a dark, smooth voice, sounding happy and very monotone in a slightly confined, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'sad', 'moderate reverberation', 'slightly noisy', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly Canadian woman speaks quickly with a bright, rough voice, sounding sad and quite expressive in a moderately reverberant, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Estonian', 'neutral', 'very roomy sounding', 'very noisy', 'slightly monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An adult Estonian man speaks at a moderate speed with a dark, smooth voice, sounding neutral and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'French', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenage girl speaks quickly with a bright, smooth voice, sounding sad and very expressive in a slightly confined, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'angry', 'moderate reverberation', 'slightly noisy', 'quite monotone', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly American man speaks quickly with a dark, rough voice, sounding angry and quite monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'neutral', 'very roomy sounding', 'very noisy', 'slightly expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An adult Irish woman speaks at a moderate speed with a bright, smooth voice, sounding neutral and slightly expressive in a very noisy, roomy environment."
""",
"""Given the keywords: 
'male', 'teenager', 'dark', 'smooth', 'Dutch', 'happy', 'slightly confined sounding', 'quite clear', 'very monotone', 'moderate pitch', 'very slowly'
A valid description could be: "A Dutch teenage boy speaks very slowly with a dark, smooth voice, sounding happy and very monotone in a slightly confined, clear room."
""",
"""Given the keywords: 
'female', 'senior', 'bright', 'rough', 'Canadian', 'sad', 'moderate reverberation', 'slightly noisy', 'quite expressive', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly Canadian woman speaks quickly with a bright, rough voice, sounding sad and quite expressive in a moderately reverberant, slightly noisy environment."
""",
"""Given the keywords: 
'male', 'adult', 'dark', 'smooth', 'Estonian', 'neutral', 'very roomy sounding', 'very noisy', 'slightly monotone', 'slightly low pitch', 'moderate speed'
A valid description could be: "An adult Estonian man speaks at a moderate speed with a dark, smooth voice, sounding neutral and slightly monotone in a very noisy, roomy environment."
""",
"""Given the keywords: 
'female', 'teenager', 'bright', 'smooth', 'French', 'sad', 'slightly confined sounding', 'quite clear', 'very expressive', 'moderate pitch', 'quite fast'
A valid description could be: "A French teenage girl speaks quickly with a bright, smooth voice, sounding sad and very expressive in a slightly confined, clear environment."
""",
"""Given the keywords: 
'male', 'senior', 'dark', 'rough', 'American', 'angry', 'moderate reverberation', 'slightly noisy', 'quite monotone', 'moderate pitch', 'quite fast'
A valid description could be: "An elderly American man speaks quickly with a dark, rough voice, sounding angry and quite monotone in a moderately reverberant, slightly noisy room."
""",
"""Given the keywords: 
'female', 'adult', 'bright', 'smooth', 'Irish', 'neutral', 'very roomy sounding', 'very noisy', 'slightly expressive', 'slightly high pitch', 'moderate speed'
A valid description could be: "An adult Irish woman speaks at a moderate speed with a bright, smooth voice, sounding neutral and slightly expressive in a very noisy, roomy environment."
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
