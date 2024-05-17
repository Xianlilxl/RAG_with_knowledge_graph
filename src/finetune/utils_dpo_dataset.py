"""
@Authors:
* Xianli Li (xli@assystem.com)
"""
import os
import wandb
import torch
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList 
from typing import Dict, List, Generator
from datasets import load_dataset, load_from_disk, concatenate_datasets
from peft import PeftModel
from transformers import GenerationConfig
import json 

def get_relation(example):
    RELATION_NAMES=['country', 'place of birth', 'spouse', 'country of citizenship', 'instance of',
            'capital', 'child', 'shares border with', 'author', 'director', 'occupation',
              'founded by', 'league', 'owned by', 'genre', 'named after', 'follows',
                'headquarters location', 'cast member', 'manufacturer',
                  'located in or next to body of water', 'location', 'part of', 
                  'mouth of the watercourse', 'member of', 'sport', 'characters',
                    'participant', 'notable work', 'replaces', 'sibling', 'inception']
    relations = []
    for relation in example['relations']:
        relation_dict = {}
        relation_dict["object"] = relation['object']['surfaceform']
        relation_dict["subject"] = relation['subject']['surfaceform']
        relation_dict["predicate"] = RELATION_NAMES[relation['predicate']]
        relations.append(relation_dict)

    return relations

def get_prompt_template() -> PromptTemplate:
    prompt_template = """### Instruction:
You are an expert in data science and natural language processing (NLP).
Your task is to extract triplets from the text provided below.
A knowledge triplet is made up of 2 entities (subject and object) linked by a predicate: 
{{"Object": "", "Predicate": "", "Subject": "" }}
Multiple triplets must be in list form.
Text: {text}\n
### Response:"""
    input_variables = ["text"]
    return PromptTemplate(
        template=prompt_template,
        input_variables=input_variables,
    )

def get_prompt(test_example):
    prompt_template = get_prompt_template()

    relations_prompt = prompt_template.format(
        text=test_example["text"],
    )

    test_example["prompt"] = relations_prompt
    return test_example

def pred_to_dict(example):
    string = example["prediction"].split("\n\n### Response:")[-1]
    string = string.replace("</s>", "").replace("Relations: ", "")
    pred_dict = json.loads(string)
    example["pred_dict"] = pred_dict
    return example

def are_lists_of_dicts_identical(list1, list2):
    # Check if the lengths of the lists are the same
    if len(list1) != len(list2):
        return False

    # Iterate through each dictionary in the lists and compare them
    for dict1, dict2 in zip(list1, list2):
        if dict1 != dict2:
            return False

    # If all dictionaries are identical, return True
    return True

def nreturn_prompt_and_responses(example):
    ground_truth = get_relation(example)
    prediction = example["pred_dict"]
    if are_lists_of_dicts_identical(ground_truth, prediction):
        example["chosen"] =  json.dumps(ground_truth)
        example["rejected"] = ""
    else:
        example["chosen"] =  json.dumps(ground_truth)
        example["rejected"] = json.dumps(prediction)
    return example