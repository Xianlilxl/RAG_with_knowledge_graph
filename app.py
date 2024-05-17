import os
from langchain.embeddings.huggingface import HuggingFaceBgeEmbeddings
from llama_index import (
    ServiceContext, 
    SimpleDirectoryReader, 
    KnowledgeGraphIndex
)

from src.kg_gen.models import ZephyrEndpointLLM, mRebelLargeEndpointLLM
from src.kg_gen.connect_nebula_graph import Nebula_Graph
from llama_index import VectorStoreIndex, ServiceContext, Document
import pdfplumber
from dotenv import load_dotenv

load_dotenv()
LLM_MODEL_NAME = os.getenv('LLM_MODEL_NAME')
HOST = os.getenv('HOST')
LLM_PORT = os.getenv('LLM_PORT')
MREBEL_PORT = os.getenv('MREBEL_PORT')
endpoint_path = os.getenv('endpoint_path')
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
context_window = os.getenv('context_window')
num_output = os.getenv('num_output')
chunk_size = os.getenv('chunk_size')

NEBULA_USER=os.getenv('NEBULA_USER')
NEBULA_PASSWORD=os.getenv('NEBULA_PASSWORD')
GRAPHD_HOST=os.getenv('GRAPHD_HOST')
GRAPHD_PORT=os.getenv('GRAPHD_PORT')
space_name=os.getenv('space_name')

# define our LLM
llm = ZephyrEndpointLLM(
    host=HOST,
    port=LLM_PORT,
    endpoint_path=endpoint_path,
    context_window=context_window,
    num_output=num_output,
    model_name=LLM_MODEL_NAME
)

embed_model = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_NAME)

service_context = ServiceContext.from_defaults(
    llm=llm, 
    embed_model=embed_model,
    context_window=int(context_window), 
    chunk_size=int(chunk_size),
    num_output=int(num_output)
)

# init mrebel large
mrebel_model = mRebelLargeEndpointLLM(
    host=HOST,
    port=MREBEL_PORT,
    endpoint_path=endpoint_path,
)

# init nebula graph
ng = Nebula_Graph(
    nebula_user=NEBULA_USER,
    nebula_password=NEBULA_PASSWORD,
    graphd_host=GRAPHD_HOST,
    graphd_port=GRAPHD_PORT,
    space_name=space_name
)
storage_context = ng.connect_nebula_graph()

kg_index = KnowledgeGraphIndex.from_documents(
        SimpleDirectoryReader(path_folder).load_data(),
        storage_context=storage_context,
        service_context=service_context,
        kg_triplet_extract_fn=mrebel_model.extract_triplets,
        max_triplets_per_chunk=3,
        # space_name=space_name,
        # edge_types=edge_types,
        # rel_prop_names=rel_prop_names,
        # tags=tags,
        )
        try:
            graph_dict = ng.save_graph(as_html=True)
        except Exception as e:
            pass
        ng.close_session_pool()
        if graph_dict:
            html = "src\kg_gen\example.html"
