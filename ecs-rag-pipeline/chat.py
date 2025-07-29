import boto3
import os
import json
import re
import logging

from typing import Iterable, List, Tuple, Set, Any
from pydantic.v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock
from botocore.config import Config
from dotenv import load_dotenv 
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
def getenv(key: str, default: str) -> str:
    return os.getenv(key, default)


os.environ['AWS_BEARER_TOKEN_BEDROCK'] = getenv('BEDROCK_API_KEY', '')

VECTOR_DIR = getenv("VECTOR_DIR", "./data_source/opensearch_vector_store")
AWS_REGION = getenv("AWS_REGION", "ap-northeast-2")
PROFILE_NAME = getenv("PROFILE_NAME", "default")
COLLECTION_NAME = getenv("COLLECTION_NAME", "work_instructions")
MODEL_IDS = {
    "titan_embedding_v2": "amazon.titan-embed-text-v2:0",
    "claude_3_5_sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude_3_5_haiku": "anthropic.claude-3-haiku-20240307-v1:0",
}

# AWS clients / Embedding model
_session = (
    boto3.Session(profile_name=PROFILE_NAME, region_name=AWS_REGION)
    if PROFILE_NAME
    else boto3.Session(region_name=AWS_REGION)
)

_bedrock_client = _session.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    config=Config( retries={"max_attempts": 30})
)
EMBEDDER = BedrockEmbeddings(model_id=MODEL_IDS["titan_embedding_v2"], client=_bedrock_client)

# Fucntions
def get_chat(model:str = "claude_3_5_haiku"):
    bedrock_region =  AWS_REGION
    modelId = MODEL_IDS[model]
    maxOutputTokens = 5120 # 4k
    logging.info(f'bedrock_region: {bedrock_region}, modelId: {modelId}, maxOutputTokens: {maxOutputTokens}')

    STOP_SEQUENCE = "\n\nHuman:" 
                          
    parameters = {
        # "max_tokens":maxOutputTokens,     
        "temperature":0.1,
        "top_k":250,
        "top_p":0.9,
        "stop_sequences": [STOP_SEQUENCE]
    }

    return ChatBedrock(
                    client=_bedrock_client,
                    region_name=AWS_REGION,
                    model_id=modelId,
                    model_kwargs=parameters,
                )

def build_or_load_chroma(persist_directory: str = VECTOR_DIR):
    return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=persist_directory,
            embedding_function=EMBEDDER,
        )

contentList = []
def check_duplication(docs):
    global contentList
    length_original = len(docs)
    
    updated_docs = []
    logging.info(f"length of relevant_docs: {len(docs)}")
    for doc in docs:            
        if doc[0].page_content in contentList:
            logging.info(f"duplicated")
            continue
        contentList.append(doc[0].page_content)
        updated_docs.append(doc)          
    length_updated_docs = len(updated_docs)

    if length_original == length_updated_docs:
        logging.info(f"no duplication")
    else:
        logging.info(f"length of updated relevant_docs: {length_updated_docs}")
    
    return updated_docs

def check_duplication(
    docs: Iterable[Tuple[Any, ...]],
    seen_contents: Set[str] | None = None,
) -> List[Tuple[Any, ...]]:

    if seen_contents is None:
        seen_contents = set()

    original_len = len(tuple(docs))  # materialize once if docs is a generator
    unique_docs: List[Tuple[Any, ...]] = []

    logging.info("length of relevant_docs: %d", original_len)

    for doc_tuple in docs:
        page_text = doc_tuple[0].page_content
        if page_text in seen_contents:
            logging.debug("duplicated")
            continue

        seen_contents.add(page_text)
        unique_docs.append(doc_tuple)

    if original_len == len(unique_docs):
        logging.info("no duplication")
    else:
        logging.info("length of updated relevant_docs: %d", len(unique_docs))

    return unique_docs