"""
@Authors:
* Xianli Li (xli@assystem.com)
This script is used to create dataset for DPO training
"""

import os
import wandb
import torch
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList 
from typing import Dict, List, Generator
from datasets import load_dataset, load_from_disk, concatenate_datasets
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import GenerationConfig
import json 

from utils_dpo_dataset import get_relation, get_prompt, pred_to_dict, return_prompt_and_responses

BASE_MODEL = "mistralai/Mistral-7B-v0.1"
QUANTIZATION_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

class StopGenerationCriteria(StoppingCriteria):
    def __init__(
            self, 
            stop_words: List[str], 
            tokenizer: AutoTokenizer, 
            device: torch.device
            ) -> None:
        stop_words = [' ' + stop_word for stop_word in stop_words]
        stop_token_ids = [tokenizer(t, add_special_tokens=False)['input_ids'][1:] for t in stop_words]
        self.stop_token_ids = [
            torch.LongTensor(x).to(device) for x in stop_token_ids
        ]

    def __call__(
            self, 
            input_ids: torch.LongTensor, 
            scores: torch.FloatTensor, 
            **kwargs
            ) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

class Model:
    def __init__(self, checkpoint_dir: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("THE DEVICE INFERENCE IS RUNNING ON IS: ", self.device)
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stopping_criteria = None
        self.checkpoint_dir = checkpoint_dir
    
    def get_checkpoint_dir(self):
        run = wandb.init()
        checkpoint = run.use_artifact(self.wandb_checkpoint_name, type='model')
        checkpoint_dir = checkpoint.download()
        return checkpoint_dir

    def load(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,  # Mistral, same as before
            quantization_config=QUANTIZATION_CONFIG,  # Same quantization config as before
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # load and merge the checkpoint with base model
        self.ft_model = PeftModel.from_pretrained(self.base_model, self.checkpoint_dir)
        self.ft_model.eval()

        self.gen_cfg = GenerationConfig.from_model_config(self.ft_model.config)
        self.gen_cfg.max_new_tokens = 1024
        self.gen_cfg.temperature = 0.5
        self.gen_cfg.num_return_sequences = 1
        self.gen_cfg.use_cache = True
        self.gen_cfg.min_length = 1

    def predict(self, request: Dict) -> Dict | Generator:
        with torch.no_grad():
            prompt = request.pop("prompt")
            stop_words = []
            if "stop" in request:
                stop_words = request.pop("stop")
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_output = self.ft_model.generate(
                input_ids=input_ids,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria = StoppingCriteriaList(
                    [StopGenerationCriteria(
                        stop_words,
                        self.tokenizer,
                        self.device
                        )]),
                generation_config=self.gen_cfg,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=256
            )
            outputs = []
            for seq in generation_output.sequences:
                output = self.tokenizer.decode(seq)
                outputs.append(output)

            return "\n".join(outputs)

pred_dataset_path = "./datasets/redfm_pred_valid_dataset"
if not os.path.exists(pred_dataset_path):
    def get_prediction(test_example):
        output = ft_model.predict(request={
            "prompt": test_example["prompt"],
            "temperature": 0.1, 
            "max_new_tokens": 1024,
        })
        test_example["prediction"] = output
        return test_example
    
    checkpoint_dir = "./checkpoints/solar-disco-79/checkpoint-4grkql3s:v10"
    ft_model = Model(checkpoint_dir=checkpoint_dir)
    ft_model.load()
    train_fr_dataset = load_dataset("Babelscape/REDFM", language="fr", split="validation")
    train_en_dataset = load_dataset("Babelscape/REDFM", language="en", split="validation")
    assert train_fr_dataset.features.type == train_en_dataset.features.type
    sliced_dataset_en = train_en_dataset.shuffle(seed=42).select(
        range(train_fr_dataset.num_rows)
    )
    concat_dataset = concatenate_datasets(
        [sliced_dataset_en, train_fr_dataset]
    ).shuffle(seed=80)
    # get prompts
    prompt_dataset = concat_dataset.map(get_prompt)
    # use merged model to generate responses
    pred_dataset = prompt_dataset.map(get_prediction)
    pred_dataset.save_to_disk("./datasets/redfm_pred_valid_dataset")
else:
    pred_dataset = load_from_disk(pred_dataset_path)

# parse the prediction into list of dictionaries
pred_dict_dataset = pred_dataset.map(pred_to_dict)
dpo_dataset = pred_dict_dataset.map(return_prompt_and_responses)
dpo_dataset = dpo_dataset.filter(lambda example: example["rejected"]!="")
dpo_dataset = dpo_dataset.remove_columns(['docid', 'title', 'uri', 'text', 'entities', 'relations', 'prediction', 'pred_dict'])
# dpo_dataset.save_to_disk("./datasets/dpo_dataset")

dpo_dataset.save_to_disk("./datasets/dpo_valid_dataset")
