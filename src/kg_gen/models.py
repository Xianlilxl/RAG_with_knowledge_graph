"""
@Authors:
* Xianli Li (xli@assystem.com)
This script defines the endpoint-based classes for Zephyr and mrebel-large
Also the customized method to define cutom node parser
"""
import requests
from typing import Optional, List, Any
from transformers import pipeline, AutoTokenizer

from llama_index.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.bridge.pydantic import Field
from llama_index.llms.base import llm_completion_callback
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.node_parser.interface import NodeParser
from llama_index.text_splitter.types import TextSplitter
from llama_index.text_splitter.sentence_splitter import SentenceSplitter
from llama_index.callbacks.base import CallbackManager


class mRebelLargeEndpointLLM:
    """
    Class for interacting with the mRebel Large Language Model (LLM) via an endpoint.

    Attributes:
        host (str): The hostname of the mRebel endpoint.
        port (str): The port number on which the mRebel endpoint is running.
        endpoint_path (str): The specific path for the endpoint.

    Methods:
        extract_triplets(input_text):
            Extract triplets from the mRebel Large Language Model (LLM) based on the input text.

            Parameters:
                input_text (str): The input text for which triplets are to be extracted.

            Returns:
                list of tuple: A list of triplets extracted from the input text.
    """
    def __init__(self, host: str = None, port: str = None, endpoint_path: str = None):
        self.host = host
        self.port = port
        self.endpoint_path = endpoint_path

    def extract_triplets(self, input_text):
        """
        Extract triplets from the mRebel Large Language Model (LLM) based on the input text.
        Note : the port should be open before 
        Parameters:
            input_text (str): The input text for which triplets are to be extracted.

        Returns:
            list of tuple: A list of triplets extracted from the input text.
        """
        MREBEL_ENDPOINT = f"http://{self.host}:{self.port}{self.endpoint_path}"
        res = requests.post(MREBEL_ENDPOINT, json=input_text)
        triplets_dict = res.json()
        triplets_dict = [tuple(triplet) for triplet in triplets_dict]
        return triplets_dict


class ZephyrEndpointLLM(CustomLLM):
    """
    ZephyrEndpointLLM class represents an endpoint-based Language Model (LLM) using the Zephyr model.

    Attributes:
        host (str): The hostname of the Zephyr endpoint.
        port (str): The port number on which the Zephyr endpoint is running.
        endpoint_path (str): The specific path for the Zephyr endpoint.
        context_window (int): The context window size for language model processing. Default is 2048.
        num_output (int): The number of output tokens expected from the language model. Default is 256.
        model_name (str): The name of the Zephyr model. Default is "HuggingFaceH4/zephyr-7b-beta".

    Methods:
        metadata() -> LLMMetadata:
            Get metadata information about the Zephyr Language Model.

        complete(prompt: str, stop: List[str] = [], temperature: float = 0.5, 
                 max_new_tokens: int = 1024, **kwargs: Any) -> CompletionResponse:
            Generate a completion for the given prompt using the Zephyr model.

        stream_complete(prompt: str, **kwargs: Any) -> CompletionResponseGen:
            [Not Implemented] Streaming completion using the Zephyr model.
    """
    host: str = None
    port: str = None
    endpoint_path: str = None
    context_window: int = 2048
    num_output: int = 256
    model_name: str = "HuggingFaceH4/zephyr-7b-beta"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = [],
        temperature: float = 0.5,
        max_new_tokens: int = 1024,
        **kwargs: Any,
    ) -> CompletionResponse:
        """
        Generate a completion for the given prompt using the Zephyr model.

        Parameters:
            prompt (str): The input prompt for completion.
            stop (List[str], optional): List of stop words to control generation.
            temperature (float): Sampling temperature for text generation.
            max_new_tokens (int): Maximum number of tokens to generate.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            CompletionResponse: The generated completion response.
        """
        data = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop or [],
        }
        try:
            response = requests.post(
                f"http://{self.host}:{self.port}{self.endpoint_path}", json=data
            )
            if response.status_code == 200:
                text = dict(response.json())["data"]["generated_text"]
            else:
                raise ValueError(
                    f"The response status code was: {response.status_code}, "
                    "expected: 200"
                )
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        raise NotImplementedError()


def get_custom_text_splitter(
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_name: str,
    callback_manager: CallbackManager = None,
) -> TextSplitter:
    """
    Get a custom text splitter for processing text data.

    Parameters:
        chunk_size (int): Size of each processing chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        tokenizer_name (str): Name of the pre-trained tokenizer.
        callback_manager (CallbackManager, optional): Callback manager for handling callbacks.

    Returns:
        TextSplitter: An instance of the custom text splitter.
    """
    paragraph_separator = "\n"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator=paragraph_separator,
        callback_manager=callback_manager,
        tokenizer=tokenizer,
    )


def get_custom_node_parser(
    chunk_size: int, 
    chunk_overlap: int, 
    tokenizer_name: str
) -> NodeParser:
    """
    Get a custom node parser for processing structured data.

    Parameters:
        chunk_size (int): Size of each processing chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        tokenizer_name (str): Name of the pre-trained tokenizer.

    Returns:
        NodeParser: An instance of the custom node parser.
    """
    text_splitter = get_custom_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        callback_manager=CallbackManager([]),
        tokenizer_name=tokenizer_name
    )
    return SimpleNodeParser.from_defaults(
        # chunk_size=chunk_size,
        # chunk_overlap=chunk_overlap,
        # callback_manager=CallbackManager([]),
        text_splitter=text_splitter,
    )
