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
from datasets import DatasetDict, load_dataset, load_from_disk
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
    push_to_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to push the processed dataset to the Hub."},
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
    dataset_cache_dir: str = field(
        default=None,
        metadata={"help": "The cache dir of the dataset to use (saved dataset)"},
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
    overwrite_output_dir: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the content of the output directory each time the script is run."},
    )
    save_steps: Optional[int] = field(
        default=500,
        metadata={"help": "Save the generated prompts every save_steps."},
    )


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
2. Age (e.g., teenager, young adult, etc.)
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
5. You can combine several keywords, e.g. if noise is 'very noisy' and reverberation is 'very roomy sounding,' describe it as a 'very bad recording.' If noise is 'very clear' and reverberation is 'very confined sounding,' describe it as a 'very good recording.'
6. Avoid repeating the same structure for consecutive descriptions. Explore different ways to convey the tone, pace, emotion, and other characteristics.
7. You can drop some of the keywords for diversity.
8. Return only the generated description.

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
         'male', 'middle-aged adult', 'dark', 'rough', 'angry', 'very roomy sounding', 'quite noisy', 'very monotone', 'low pitch', 'very slowly',
    A valid description could be: "A middle-aged man speaks in a dark, rough voice with a low pitch. His tone is monotone and angry, the delivery slow, and the sound environment very noisy and roomy."
    """,
    """Given the keywords:
         'female', 'child', 'bright', 'smooth', 'happy', 'slightly confined sounding', 'very clear', 'very expressive', 'high pitch', 'fast pace',
    A valid description could be: "A cheerful child speaks quickly with a high-pitched, bright voice. Her tone is expressive and smooth, the sound clear and slightly confined."
    """,
    """Given the keywords:
         'male', 'elderly', 'neutral', 'rough', 'sad', 'moderately roomy', 'quite clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "An elderly man speaks slowly in a low-pitched, rough voice. His tone is neutral yet sad, the speech clear but somewhat monotone in a moderately roomy setting."
    """,
    """Given the keywords:
         'female', 'young adult', 'slightly dark', 'smooth', 'angry', 'confined sounding', 'very clear', 'very expressive', 'medium pitch', 'average pace',
    A valid description could be: "A young woman delivers her speech with a smooth and slightly dark voice, her anger evident. The medium-pitched speech is clear and expressive, with a confined sound and average pace."
    """,
    """Given the keywords:
         'male', 'teenager', 'neutral', 'smooth', 'happy', 'slightly roomy', 'quite noisy', 'expressive', 'medium pitch', 'fast pace',
    A valid description could be: "A teenage boy speaks quickly in a medium-pitched, smooth voice. His tone is expressive and happy, though the slightly roomy and noisy sound detracts from its quality."
    """,
    """Given the keywords:
         'female', 'middle-aged adult', 'slightly bright', 'rough', 'neutral', 'confined sounding', 'very clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "A middle-aged woman speaks with a slightly bright tone and rough voice, her delivery slow and monotone. The sound is confined and clear, with a low pitch adding depth."
    """,
    """Given the keywords:
         'male', 'young adult', 'dark', 'smooth', 'neutral', 'very roomy sounding', 'very noisy', 'somewhat monotone', 'low pitch', 'very slowly',
    A valid description could be: "A young man speaks in a slow, monotone manner, his dark and smooth voice resonating with a low pitch. The recording is noisy and spacious, which affects the clarity."
    """,
     """Given the keywords:
         'female', 'teenager', 'slightly bright', 'smooth', 'neutral', 'slightly confined sounding', 'very clear', 'somewhat expressive', 'medium pitch', 'average pace',
    A valid description could be: "The voice of a teenage girl carries a neutral tone and a smooth, slightly bright timbre. With a medium pitch and clear sound, her speech feels expressive yet moderately paced."
    """,
    """Given the keywords:
         'male', 'middle-aged adult', 'dark', 'rough', 'angry', 'very roomy sounding', 'quite noisy', 'very monotone', 'low pitch', 'very slowly',
    A valid description could be: "Speaking with anger, the middle-aged man’s rough and dark voice resonates in a noisy, spacious room. His monotone delivery is slow, emphasizing the low pitch of his speech."
    """,
    """Given the keywords:
         'female', 'child', 'bright', 'smooth', 'happy', 'slightly confined sounding', 'very clear', 'very expressive', 'high pitch', 'fast pace',
    A valid description could be: "Joy radiates from the voice of a child, whose high-pitched, smooth timbre is both expressive and bright. The fast-paced words are captured clearly in a slightly confined acoustic space."
    """,
    """Given the keywords:
         'male', 'elderly', 'neutral', 'rough', 'sad', 'moderately roomy', 'quite clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "In a moderately roomy environment, an elderly man’s rough, low-pitched voice conveys a sense of sadness. His slow and somewhat monotone delivery is accompanied by clear sound quality."
    """,
    """Given the keywords:
         'female', 'young adult', 'slightly dark', 'smooth', 'angry', 'confined sounding', 'very clear', 'very expressive', 'medium pitch', 'average pace',
    A valid description could be: "Anger is evident in the speech of a young woman, whose smooth, slightly dark timbre carries a medium pitch. Her words, clear and expressive, are delivered with an average rhythm in a confined space."
    """,
    """Given the keywords:
         'male', 'teenager', 'neutral', 'smooth', 'happy', 'slightly roomy', 'quite noisy', 'expressive', 'medium pitch', 'fast pace',
    A valid description could be: "Words spill rapidly from the voice of a happy teenage boy. His smooth, medium-pitched tone is expressive, though the slightly roomy and noisy environment impacts clarity."
    """,
    """Given the keywords:
         'female', 'middle-aged adult', 'slightly bright', 'rough', 'neutral', 'confined sounding', 'very clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "Neutral in tone, the rough, slightly bright voice of a middle-aged woman reflects a low pitch. Her speech, delivered slowly, sounds confined yet clear, with little variation in expressiveness."
    """,
    """Given the keywords:
         'male', 'young adult', 'dark', 'smooth', 'neutral', 'very roomy sounding', 'very noisy', 'somewhat monotone', 'low pitch', 'very slowly',
    A valid description could be: "Resonating in a very roomy, noisy environment, the young man’s dark, smooth voice reflects neutrality. The monotone speech is slow and low-pitched, adding to its subdued delivery."
    """,
    """Given the keywords:
         'female', 'teenager', 'slightly bright', 'smooth', 'neutral', 'slightly confined sounding', 'very clear', 'somewhat expressive', 'medium pitch', 'average pace',
    A valid description could be: "Her voice, slightly bright and smooth, flows with a neutral tone and a medium pitch. The sound is very clear and somewhat expressive, resonating in a slightly confined space at an average pace."
    """,
    """Given the keywords:
         'male', 'middle-aged adult', 'dark', 'rough', 'angry', 'very roomy sounding', 'quite noisy', 'very monotone', 'low pitch', 'very slowly',
    A valid description could be: "Dark and rough in texture, his voice projects anger. The low-pitched monotone resonates in a very noisy, spacious room, with words delivered at an unhurried pace."
    """,
    """Given the keywords:
         'female', 'child', 'bright', 'smooth', 'happy', 'slightly confined sounding', 'very clear', 'very expressive', 'high pitch', 'fast pace',
    A valid description could be: "Radiating joy, the child speaks with a high-pitched, bright, and smooth voice. Her words, fast and expressive, are captured clearly in a slightly confined sound environment."
    """,
    """Given the keywords:
         'male', 'elderly', 'neutral', 'rough', 'sad', 'moderately roomy', 'quite clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "In a moderately roomy setting, his rough voice reflects a sense of sadness. Each low-pitched, neutral word emerges clearly, though the tone is somewhat monotone and delivered slowly."
    """,
    """Given the keywords:
         'female', 'young adult', 'slightly dark', 'smooth', 'angry', 'confined sounding', 'very clear', 'very expressive', 'medium pitch', 'average pace',
    A valid description could be: "Anger weaves through her slightly dark, smooth voice as she speaks at an average pace. The medium-pitched tone is clear and expressive, reverberating in a confined acoustic space."
    """,
    """Given the keywords:
         'male', 'teenager', 'neutral', 'smooth', 'happy', 'slightly roomy', 'quite noisy', 'expressive', 'medium pitch', 'fast pace',
    A valid description could be: "His medium-pitched voice, both smooth and expressive, conveys happiness. Despite the slightly roomy and noisy surroundings, the teenager's words are delivered rapidly and with enthusiasm."
    """,
    """Given the keywords:
         'female', 'middle-aged adult', 'slightly bright', 'rough', 'neutral', 'confined sounding', 'very clear', 'somewhat monotone', 'low pitch', 'slow pace',
    A valid description could be: "A neutral tone defines her slightly bright yet rough voice, with each word emerging clearly in a confined space. The low pitch and slow pace add to the monotony of her delivery."
    """,
    """Given the keywords:
         'male', 'young adult', 'dark', 'smooth', 'neutral', 'very roomy sounding', 'very noisy', 'somewhat monotone', 'low pitch', 'very slowly',
    A valid description could be: "Low-pitched and smooth, his neutral voice struggles to cut through the very noisy, spacious environment. The monotone delivery is slow, with a dark timbre that gives the speech a somber quality."
    """,
    """Given the keywords:
         'female', 'child', 'bright', 'smooth', 'excited', 'confined sounding', 'quite clear', 'very expressive', 'high pitch', 'fast pace',
    A valid description could be: "Excitement bubbles through her high-pitched, smooth voice, making the bright timbre even more vibrant. The confined space amplifies her expressive delivery, which races forward at a rapid pace."
    """,
    """Given the keywords:
         'male', 'middle-aged adult', 'slightly dark', 'rough', 'neutral', 'moderately confined sounding', 'clear', 'monotone', 'medium pitch', 'average pace',
    A valid description could be: "Each word emerges with rough edges and a slightly dark timbre, reflecting a neutral tone. The moderately confined acoustic adds clarity to his medium-pitched voice, though it remains monotone and steady-paced."
    """,
    """Given the keywords:
         'female', 'teenager', 'bright', 'smooth', 'happy', 'slightly roomy sounding', 'very clear', 'expressive', 'high pitch', 'fast pace',
    A valid description could be: "Her happy voice, bright and smooth, fills the slightly roomy space with high-pitched clarity. The teenager's expressive tone is carried briskly, reflecting a sense of vitality."
    """,
    """Given the keywords:
         'male', 'elderly', 'neutral', 'rough', 'sad', 'very roomy sounding', 'quite noisy', 'somewhat monotone', 'low pitch', 'very slowly',
    A valid description could be: "Resonating within a noisy, expansive room, his rough, low-pitched voice conveys sadness. The monotone speech feels heavy, delivered at a very slow pace and devoid of variation."
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


    # 3. Load annotated dataset
    logger.info("*** Load annotated dataset ***")
    
    with accelerator.local_main_process_first():
        raw_datasets = load_from_disk(data_args.dataset_cache_dir)

    raw_datasets_features = set(raw_datasets.features)

    # TODO(SG): add accent
    EXPECTED_COLUMNS = {'gender', 'age', 'brightness', 'smoothness', 'emotion', 'reverberation', 'noise', 'speech_monotony', 'pitch', 'speaking_rate'}
    
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


    data_loader = DataLoader(
        vectorized_datasets,
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

    all_generated_ids, cur_step = get_last_checkpoint(data_args.output_dir)

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

    vectorized_datasets = vectorized_datasets.add_column("generated_ids", all_generated_ids)

    if accelerator.is_main_process:
        vectorized_datasets = vectorized_datasets.map(
            postprocess_dataset,
            num_proc=data_args.preprocessing_num_workers,
            desc="Postprocessing dataset",
            remove_columns=["input_ids", "generated_ids"],
        )
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        vectorized_datasets.save_to_disk(data_args.output_dir)
        
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
