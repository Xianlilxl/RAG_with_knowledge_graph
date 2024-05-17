# This is a modified version of TRL's `SFTTrainer` example (https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py),
# adapted to run with DeepSpeed ZeRO-3 and Mistral-7B-V1.0. The settings below were run on 1 node of 8 x A100 (80GB) GPUs.
#
# Usage:
#   - Install the latest transformers & accelerate versions: `pip install -U transformers accelerate`
#   - Install deepspeed: `pip install deepspeed==0.9.5`
#   - Install TRL from main: pip install git+https://github.com/huggingface/trl.git
#   - Clone the repo: git clone github.com/huggingface/trl.git
#   - Copy this Gist into trl/examples/scripts
#   - Run from root of trl repo with: accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml --gradient_accumulation_steps 8 examples/scripts/sft_trainer.py
"""
@Authors:
* Xianli Li (xli@assystem.com)
"""
from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import LoraConfig
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import os, wandb

from create_dataset import (
    load_one_dataset,
    load_and_concatenate_datasets,
    redo_train_test_split,
)

os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["WANDB_BASE_URL"] = "https://api.wandb.ai"
wandb.init(project="digital_safety", entity="xianli")

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """

    model_name: Optional[str] = field(
        default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"}
    )
    dataset_name: Optional[str] = field(
        default="stingning/ultrachat", metadata={"help": "the dataset name"}
    )
    dataset_language: Optional[str] = field(
        default="fr", metadata={"help": "the dataset language"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    log_with: Optional[str] = field(
        default="wandb", metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=2.0e-5, metadata={"help": "the learning rate"}
    )
    batch_size: Optional[int] = field(default=8, metadata={"help": "the batch size"})
    seq_length: Optional[int] = field(
        default=1024, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=False, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=5, metadata={"help": "the number of logging steps"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=1000,
        metadata={"help": "Number of updates steps before two checkpoint saves"},
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    neftune_noise_alpha: Optional[float] = field(
        default=5.0, metadata={"help": "activate NEFTune noise embeddings"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Step 1: Load the dataset and generate prompt
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token
# if the user select only one language, we directly load the model
# In this step we concatenate the previous split to avoid the uneven distribution of train and test dataset
# after the following filter operation
split_dataset_path = "./datasets/SREDFM-dataset-" + str(script_args.seq_length)
if not os.path.exists(split_dataset_path):
    if script_args.dataset_language == "fr" or script_args.dataset_language == "en":
        dataset = load_one_dataset(
            script_args.dataset_name, dataset_language=script_args.dataset_language
        )
    # if the user select more than one language, we concatenate 2 datasets
    else:
        dataset = load_and_concatenate_datasets(script_args.dataset_name)

    # We select only examples shorter than the given sequence length
    # to avoid information lost
    dataset = dataset.filter(
        lambda example: len(example["text"]) < script_args.seq_length, 
        num_proc=4,
        load_from_cache_file=False
    )
    # redo the split
    split_dataset = redo_train_test_split(dataset)
    # save the preprocessed dataset to disk
    split_dataset_path = "./datasets/SREDFM-dataset-" + str(script_args.seq_length)
    split_dataset.save_to_disk(split_dataset_path)
else:
    split_dataset = load_from_disk(split_dataset_path)

# save the dataset to wandb artifact
dataset_artifact = wandb.Artifact(
    "SREDFM-dataset",
    type="dataset",
    description="filtered dataset by max_seq_length of " + str(script_args.seq_length),
)
dataset_artifact.add_dir(split_dataset_path)  # Adds multiple files to artifact
wandb.log_artifact(dataset_artifact)

# remove unnecessary columns 
unnecessary_columns = ["docid", "title", "uri", "entities", "relations"]
split_dataset["train"] = split_dataset["train"].remove_columns(unnecessary_columns)
split_dataset["validation"] = split_dataset["validation"].remove_columns(unnecessary_columns)
print(split_dataset)
# # Step 2: Load the model
if script_args.load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=script_args.load_in_4bit,
        bnb_4bit_use_double_quant=script_args.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    # Copy the model to each device
    device_map = "auto"
    torch_dtype = torch.bfloat16
else:
    device_map = None
    quantization_config = None
    torch_dtype = None

# load the model with pre-defined config
model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    use_auth_token=script_args.use_auth_token,
)

# Step 3: Define the training arguments
training_args = TrainingArguments(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=True,
    learning_rate=script_args.learning_rate,
    logging_steps=script_args.logging_steps,
    num_train_epochs=script_args.num_train_epochs,
    max_steps=script_args.max_steps,
    report_to=script_args.log_with,
    save_steps=script_args.save_steps,
    eval_steps=script_args.save_steps,
    # save_total_limit=script_args.save_total_limit,
    bf16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    # evaluation_strategy="epoch",
    evaluation_strategy="steps",
    do_eval=True,
    logging_first_step=True,
    load_best_model_at_end=True,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(
        r=script_args.peft_lora_r,
        lora_alpha=script_args.peft_lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        task_type="CAUSAL_LM",
        lora_dropout=0.1
    )
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    max_seq_length=script_args.seq_length,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["validation"],
    dataset_text_field=script_args.dataset_text_field,
    peft_config=peft_config,
    packing=True,
    # data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, 
                                                # instruction_template="Instruction:",
                                                # response_template="Response:"),
    neftune_noise_alpha=script_args.neftune_noise_alpha,
)

trainer.train()

# Step 6: Save the model
trainer.save_model(script_args.output_dir)
