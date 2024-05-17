"""
@Authors:
* Xianli Li (xli@assystem.com)
This script is used to create dataset for finetuning mistral 7b
"""
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer
from typing import Dict
import json

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


def get_relation(example: Dict) -> str:
    """
    Extracts and structures relations from a single example within a English dataset.

    Args:
        example (dict): A dictionary containing entities and relations.

    Returns:
        str: A string representation of the extracted relations.

    Example:
        Given an 'example' dictionary containing 'entities' and 'relations', this function
        extracts and structures relations, returning them as a string.

    """
    entities_ls = example["entities"]
    relations = []
    for relation in example["relations"]:
        relation_dict = {}
        object_index = relation["object"]
        relation_dict["Object"] = entities_ls[object_index]["surfaceform"]
        relation_dict["Predicate"] = relation["predicate"]
        subject_index = relation["subject"]
        relation_dict["Subject"] = entities_ls[subject_index]["surfaceform"]
        relations.append(relation_dict)

    return json.dumps(relations, ensure_ascii=False)


def get_entities(example: Dict) -> str:
    """
    Extracts and returns unique entities from a single data instance within a dataset.

    Args:
        example (dict): A dictionary containing entities information for a single data instance.

    Returns:
        str: A string representation of the unique entities.

    Example:
        Given an 'example' dictionary containing 'entities' for a single data instance within a
        dataset, this function extracts and returns the unique entities as a string.

    """

    # Extract the list of entities' surface forms without duplicates
    entities = example["entities"]
    unique_entities_surface_forms = list(
        set(entity["surfaceform"] for entity in entities)
    )

    # Convert the unique entities to a string
    unique_entities_str = ', '.join('"{}"'.format(entity.replace('"', "")) for entity in unique_entities_surface_forms)
    unique_entities_str = "["+unique_entities_str+"]"

    # Return the unique entities as a string
    return unique_entities_str


def get_prompt_template() -> PromptTemplate:
    """
    Creates a prompt template for extracting knowledge triplets from a given text.

    Returns:
        PromptTemplate: A template for extracting triplets from text.

    Example:
        This function generates a prompt template that can be used for extracting
        knowledge triplets from a text. The template includes placeholders for the text,
        entities, and relations.
    """
    prompt_template = """Here are two instructions on NLP task, each accompanied by a corresponding text. Extract entities and triplets from the provided texts based on the given instructions.
### Instruction:
Extract entities from the given text. Entities are the subject and object of a sentence.
### Text:
{text}\n
### Response:
Entities: {entities}{eos_token}\n
### Instruction:
Now based on the entities that you extracted before, you should extract all knowledge triplets.
A knowledge triplet is made up of 2 entities (subject and object) linked by a predicate: 
{{"Object": "", "Predicate": "", "Subject": "" }}
Entities can be related to many other entities.
Multiple triplets must be in list form.\n
### Response:
Relations: {relations}{eos_token}\n
"""
    input_variables = ["text", "eos_token", "entities", "relations"]
    return PromptTemplate(
        template=prompt_template,
        input_variables=input_variables,
    )
# def get_fr_prompt_template() -> PromptTemplate:
#     """
#     Creates a prompt template for extracting knowledge triplets from a given French text.

#     Returns:
#         PromptTemplate: A template for extracting triplets from text.

#     Example:
#         This function generates a prompt template that can be used for extracting
#         knowledge triplets from a text. The template includes placeholders for the text,
#         entities, and relations.
#     """
#     fr_prompt_template = """### Instruction:
# Vous êtes un expert en data science et en traitement du langage naturel(NLP).
# Votre tâche consiste à extraire les triplets du TEXTE fourni ci-dessous.
# Les entité s'agit du sujet et de l'objet d'une phrase, la liste d'entités doit être sous forme:
# ['entité1', 'entité2', 'entité3', ...]
# Un triplet de connaissances est constitué de 2 entités (sujet et objet) liées par un prédicat : 
# {{"Objet": "","Prédicat": "", "Sujet": "" }}
# Les triples multiples doivent être sous forme de liste.\n
# Text: {text}\n
# ### Response:
# Entities: {entities}\n
# Relations: {relations}{eos_token}\n
# """
#     input_variables = ["text", "eos_token", "entities", "relations"]

#     return PromptTemplate(
#         template=fr_prompt_template,
#         input_variables=input_variables,
#     )

# def generate_fr_base_prompt(example: Dict) -> Dict:
#     """
#     Generates a prompt for extracting knowledge triplets from a given example within a French dataset.

#     Args:
#         example (dict): A dictionary containing data for extraction.

#     Returns:
#         dict: A dictionary containing the generated extraction prompt.

#     Example:
#         This function generates a prompt for extracting knowledge triplets from a given 'example'.
#     """
#     # Get the French prompt template
#     prompt_template = get_fr_prompt_template()

#     # Retrieve the EOS token from the tokenizer
#     eos_token = tokenizer.eos_token

#     # Get the entities and relations from the French example
#     entities = get_entities(example)
#     relations = get_relation(example)

#     # Create the full prompt by filling in the template
#     full_prompt = prompt_template.format(
#         text=example["text"],
#         eos_token=eos_token,
#         entities=entities,
#         relations=relations,
#     )
#     example["text"] = full_prompt
#     return example


def generate_base_prompt(example: Dict) -> Dict:
    """
    Generates a prompt for extracting knowledge triplets from a given example within a English dataset.

    Args:
        example (dict): A dictionary containing data for extraction.

    Returns:
        dict: A dictionary containing the generated extraction prompt.

    Example:
        This function generates a prompt for extracting knowledge triplets from a given 'example'.
    """
    # Get the English prompt template
    prompt_template = get_prompt_template()

    # Retrieve the EOS token from the tokenizer
    # bos_token = tokenizer.bos_token
    eos_token = tokenizer.eos_token

    # Get the entities and relations from the English example
    entities = get_entities(example)
    relations = get_relation(example)

    # Create the full prompt by filling in the template
    full_prompt = prompt_template.format(
        text=example["text"],
        eos_token=eos_token,
        entities=entities,
        relations=relations,
    )

    # Return the full prompt as a dictionary
    example["text"] = full_prompt
    return example


def redo_train_test_split(dataset: Dataset) -> DatasetDict:
    """
    Redo the train-test split for a given dataset.

    Args:
        dataset (Dataset): The dataset to be split.

    Returns:
        DatasetDict: A dictionary containing the split train, test, and validation datasets.

    Example:
        This function shuffles the input dataset and re-splits it into train, test, and validation subsets.
        It returns a DatasetDict with the split datasets, suitable for various machine learning tasks.
    """
    # shuffle the dataset before the new split
    shuffled_dataset = dataset.shuffle(seed=80)
    train_test_dataset = shuffled_dataset.train_test_split(test_size=0.01, seed=42)

    # Split the 0.01% test + valid in half test, half valid
    test_valid = train_test_dataset["test"].train_test_split(test_size=0.5, seed=42)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict(
        {
            "train": train_test_dataset["train"],
            "test": test_valid["test"],
            "validation": test_valid["train"],
        }
    )
    return train_test_valid_dataset


def load_one_dataset(dataset_name: str, dataset_language: str = "fr") -> Dataset:
    """
    Loads and processes a dataset, including generating prompts and concatenating splits.

    Args:
        dataset_name (str): The name of the dataset to load.
        dataset_language (str, optional): The language of the dataset, default is "fr".
            For now it only supports French and English dataset.

    Returns:
        Dataset: The processed dataset with generated prompts and concatenated splits.

    Example:
        This function loads a dataset with the specified name and language, generates prompts based on
        the language, and concatenates the train, test, and validation splits into a single dataset.
    """
    dataset = load_dataset(dataset_name, language=dataset_language)
    # if dataset_language=="en":
    dataset = dataset.map(
        generate_base_prompt,
        num_proc=4,
        load_from_cache_file=False
    )
    # else:
    #     dataset = dataset.map(
    #         generate_fr_base_prompt,
    #         num_proc=4,
    #         load_from_cache_file=False
    #     )
    concat_dataset = concatenate_datasets(
        [dataset["train"], dataset["test"], dataset["validation"]]
    )
    return concat_dataset


def load_and_concatenate_datasets(dataset_name: str) -> Dataset:
    """
    Loads and processes two datasets in different languages, generating prompts and concatenating them.
    For now it only supports French and English dataset.
    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        Dataset: The processed dataset with generated prompts and concatenated data.

    Example:
        This function loads two datasets in different languages, generates prompts for each language,
        matches the dataset sizes, and then concatenates them into a single dataset for downstream tasks.
    """
    # Load and concatenate the French dataset splits
    dataset_fr = load_one_dataset(dataset_name, dataset_language="fr")
    # Load and oncatenate the English dataset splits
    dataset_en = load_one_dataset(dataset_name, dataset_language="en")

    # Ensure that both datasets have the same features
    assert dataset_fr.features.type == dataset_en.features.type

    # Slice the english dataset to match the french dataset size
    sliced_dataset_en = dataset_en.shuffle(seed=42).select(
        range(dataset_fr.num_rows)
    )

    # Concatenate all prompt datasets and shuffle
    concat_dataset = concatenate_datasets(
        [sliced_dataset_en, dataset_fr]
    ).shuffle(seed=80)

    return concat_dataset
