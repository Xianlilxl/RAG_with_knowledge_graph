# RAG with knowledge graph
Our comprehensive solution entails the creation of a knowledge graph generated from the provided corpus. Subsequently, this knowledge graph is seamlessly integrated into a Retrieval Augmented Generation (RAG) system. This sophisticated system empowers users to interactively query and extract valuable insights from the knowledge graph, facilitating a more intuitive and efficient exploration of safety-related information.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Fine-tuning](#fine-tuning)
4. [Usage](#usage)
5. [Dependencies](#dependencies)
## Introduction

This project can be divided into two parts: 
- **knowledge graph generation** : we use generative intruction finetuned LLM and SOTA of relation extraction model (mREBEL-large) to create knowledge graph. 
- **finetuning** : we have tempted the meaning of finetuning Mistral 7B base model to improve the existing instruction finetuned LLM on relation extraction. For the moment, We use **REBEL large** for knowledge graph generation. 


## Project Structure

```
RAG with knowledge graph
├── notebooks
|   ├── finetune
│   ├── kg_gen
├── src
|   ├── finetune
|   |   ├── create_dataset.py
|   |   ├── create_dpo_dataset.py
|   |   ├── create_GT_pred_dataset.py
|   |   ├── dpo.py
|   |   ├── merge_peft_adapter.py
|   |   ├── score.py
|   |   ├── sft_trainer.py
|   |   ├── utils_dpo_dataset.py
│   ├── kg_gen
│   │   ├── connect_nebula_graph.py
│   │   ├── kg_gen_rebel.py
│   │   ├── models.py
│   └── static
├── .env
├── .gitignore
├── app.py
├── docker-compose.yml
├── README.md
└── requirements.txt
```


- **notebooks**: Jupyter notebooks used for finetuning and knowledge graph generation. They are mainly drafts, all codes are implemented in the `src` file.
- **src**: Contains the source code of the project, can be devided into two parts:
    - **finetune**: it contains all necessary scripts used to finetune the Mistral 7B base model, including dataset creation, DPO method. 
    - **kg_gen**: it contains all necessary scripts used in the `app.py` to generate automatically the knowledge graph.
- **.env**: env file containing all necessary environment variable.
- **app.py**: Main file to launch the entire program.
- **docker-compose.yml**: launchs nebula graph containers and model serving containers.
- **requirements.txt**: contains necessary python package to run the application.

## Fine-Tuning

If you need to perform fine-tuning, follow these steps:

1. Navigate to the project directory in your terminal.
    ```
    cd ./src/finetune
    ```

2. Launch the fine-tuning process by executing the following command:

    ```bash
    python sft_trainer.py --dataset_name "Babelscape/SREDFM" --dataset_language "fr+en" --seq_length 1024 \
        --load_in_4bit True --use_peft True --trust_remote_code True --output_dir "./models/mistral-KG-finetune_sft_neftune" \
        --batch_size 8 --peft_lora_r 16 --peft_lora_alpha 64 --use_auth_token False --max_steps 500 \
        --save_steps 25
    ```

   Adjust the command-line arguments based on your specific requirements.

   - `--dataset_name`: Specify the dataset name.
   - `--dataset_language`: Specify the dataset language. (3 options: fr, en, fr+en)
   - `--seq_length`: Set the sequence length.
   - `--load_in_4bit`: Toggle loading in 4-bit mode.
   - `--use_peft`: Enable PEFT (Placeholder Evaluation for Finetuning).
   - `--trust_remote_code`: Trust remote code during the process.
   - `--output_dir`: Set the output directory for the fine-tuned model.
   - `--batch_size`: Define the batch size.
   - `--peft_lora_r`: Set PEFT LoRA radius.
   - `--peft_lora_alpha`: Set PEFT LoRA alpha.
   - `--use_auth_token`: Toggle the use of authentication token.
   - `--max_steps`: Set the maximum number of training steps.
   - `--save_steps`: Set the frequency of saving checkpoints.

3. The fine-tuning process will commence, and you can monitor the progress in the console.

   Note: Adjust the parameters according to your specific dataset and requirements.

___
If you want to apply Direct Preference Optimization (DPO) to further enhance the performance of your fine-tuned model, follow these steps:

1. Ensure you have a previously fine-tuned model that you want to continue optimizing.

2. Execute this command below to create dpo dataset:
```
python create_dpo_dataset.py
```
3. Launch the DPO process by executing the dpo.py script:
```
python dpo.py
```
The dpo.py script isn't finished yet, so please refer to the following blog for further information : [Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl).

## Usage

To get started with the project, follow these steps:

1. Install the required dependencies by running the following command in your environment:

    ```bash
    pip install -r requirements.txt
    ```

2. Use Docker Compose to build and start the application:

    ```bash
    docker-compose --env-file .env up -d
    ```

   This will launch the necessary containers in the background.

3. Finally, run the application with the following command:

    ```bash
    python app.py
    ```
    After successful launch, your application will be accessible. Open your web browser and visit the provided link (usually http://localhost:8501) to interact with the application.

   Note: The specific link may vary based on your application's configuration. Check the console output for the exact URL.


## Dependencies

Ensure you have the following software and dependencies installed before running the project:

- **Python 3.10.11**: This project is designed to work with Python 3.10.11. 

- **CUDA**: CUDA is essential for GPU acceleration. Install the compatible version of CUDA. You can find CUDA downloads on the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).

- **WSL2 (Windows Subsystem for Linux 2)**: WSL2 is required to use Docker on Windows. Follow the official documentation to set up WSL2 on [Microsoft's WSL documentation](https://docs.microsoft.com/en-us/windows/wsl/install).

- **Docker**: This project utilizes Docker for containerization. 
- **Python Packages**: Install the required Python packages by running the following command in your virtual environment:

  ```bash
  pip install -r requirements.txt
