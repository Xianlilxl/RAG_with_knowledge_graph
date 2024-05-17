import os
import requests
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from IPython.display import Markdown, display
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig
)
from typing import Optional, List, Mapping, Any, Tuple
from langchain import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import (
    ServiceContext, 
    SimpleDirectoryReader, 
#     LangchainEmbedding, 
#     ListIndex,
    KnowledgeGraphIndex
)
from llama_index.callbacks import CallbackManager
from llama_index.llms import (
    CustomLLM, 
    CompletionResponse, 
    CompletionResponseGen,
    LLMMetadata,
)

from llama_index.llms.base import llm_completion_callback

from models import ZephyrEndpointLLM, extract_triplets
from connect_nebula_graph import Nebula_Graph

# define our LLM
llm = ZephyrEndpointLLM(api_endpoint="http://127.0.0.1:8080")

embed_model = HuggingFaceBgeEmbeddings(model_name="dangvantuan/sentence-camembert-large")

context_window = 2048
# set number of output tokens
num_output = 1024
chunk_size = 128

service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model=embed_model,
    context_window=context_window, 
    chunk_size=chunk_size,
    num_output=num_output
)

ng = Nebula_Graph(space_name="test")
storage_context = ng.connect_nebula_graph()

# Load the your data
documents = SimpleDirectoryReader("../../../data/").load_data()

kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    service_context=service_context,
    kg_triplet_extract_fn=extract_triplets,
    max_triplets_per_chunk=3,
    # space_name=space_name,
    # edge_types=edge_types,
    # rel_prop_names=rel_prop_names,
    # tags=tags,
)
try:
    ng.save_to_html()
except Exception as e:
    pass
ng.close_session_pool()

