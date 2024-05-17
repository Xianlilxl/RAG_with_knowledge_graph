from dataclasses import dataclass, field
from typing import Optional

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    """
    The input names representing the Adapter and Base model fine-tuned with PEFT, and the output name representing the
    merged model.
    """

    adapter_model_path: Optional[str] = field(default=None, metadata={"help": "the adapter path"})
    base_model_name: Optional[str] = field(default=None, metadata={"help": "the base model name"})
    output_name: Optional[str] = field(default=None, metadata={"help": "the merged model name"})


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
assert script_args.adapter_model_path is not None, "please provide the path of the Adapter you would like to merge"
assert script_args.base_model_name is not None, "please provide the name of the Base model"
assert script_args.output_name is not None, "please provide the output name of the merged model"

model = AutoModelForCausalLM.from_pretrained(
    script_args.base_model_name, return_dict=True, torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)

# Load the PEFT model
model = PeftModel.from_pretrained(model, script_args.adapter_model_path)
model.eval()

model = model.merge_and_unload()

model.save_pretrained(f"{script_args.output_name}")
tokenizer.save_pretrained(f"{script_args.output_name}")
# model.push_to_hub(f"{script_args.output_name}", use_temp_dir=False)