import json
import logging
import os
import pathlib
import uuid
from datetime import datetime
from typing import List

import boto3
from botocore.config import Config
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_chroma import Chroma
from dotenv import load_dotenv 
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
def getenv(key: str, default: str) -> str:
    return os.getenv(key, default)

os.environ['AWS_BEARER_TOKEN_BEDROCK'] = getenv('BEDROCK_API_KEY', '')

VECTOR_DIR = getenv("VECTOR_DIR", "./data_source/opensearch_vector_store")
PDF_DIR = getenv("PDF_DIR", "./data_source/s3_work_instruction_pdf")
COLLECTION_NAME = getenv("COLLECTION_NAME", "work_instructions")
MODEL_ID = getenv("MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_REGION = getenv("AWS_REGION", "ap-northeast-2")
PROFILE_NAME = getenv("PROFILE_NAME", "default")  
CHUNK_SIZE = int(getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(getenv("CHUNK_OVERLAP", "200"))


# AWS clients / Embedding model
_session = (
    boto3.Session(profile_name=PROFILE_NAME, region_name=AWS_REGION)
    if PROFILE_NAME
    else boto3.Session(region_name=AWS_REGION)
)

_bedrock_client = _session.client(
    "bedrock-runtime",
    config=Config(region_name=AWS_REGION, retries={"max_attempts": 3})
)

EMBEDDER = BedrockEmbeddings(model_id=MODEL_ID, client=_bedrock_client)


# Fucntions
def load_documents_from_directory(directory: str) -> List[dict]:
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyMuPDFLoader, show_progress=True)
    docs = loader.load()
    logging.info("Loaded %d PDF documents from %s", len(docs), directory)
    return docs

def load_documents_from_file(file_path: str) -> List[dict]:
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    logging.info("Loaded %d PDF documents from %s", len(docs), file_path)
    return docs

def split_documents(documents: List[dict]) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n- ", "\n", " "],
    )
    chunks = splitter.split_documents(documents)
    logging.info("Split into %d chunks", len(chunks))
    return chunks


def upsert_chunks(chunks: List[dict], vector_dir: str = VECTOR_DIR) -> int:
    """Add *chunks* to Chroma collection (creates collection on first run)."""
    vec_path = pathlib.Path(vector_dir)
    vec_path.mkdir(parents=True, exist_ok=True)

    # Create or load collection
    vectorstore = Chroma(
        persist_directory=str(vec_path),
        embedding_function=EMBEDDER,
        collection_name=COLLECTION_NAME,
    )

    existing = vectorstore.get()["ids"]
    logging.info("Collection '%s' currently holds %d vectors", COLLECTION_NAME, len(existing))

    if existing:
        vectorstore.add_documents(chunks)
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=EMBEDDER,
            persist_directory=str(vec_path),
            collection_name=COLLECTION_NAME,
        )

    # vectorstore.persist()
    total = len(vectorstore.get()["ids"])
    logging.info("Persisted collection now holds %d vectors", total)
    return total


# Lambda handler
def lambda_handler(event, context):
    logging.info("Event received: %s", json.dumps(event))

    file_name = event.get("file_name")
    if '.pdf' in file_name:
        local_pdf_path = f"./{PDF_DIR}/{file_name}"
        documents = load_documents_from_file(local_pdf_path)
    else:
        documents = load_documents_from_directory(PDF_DIR)
    chunks = split_documents(documents)
    total_vectors = upsert_chunks(chunks)

    body = {
        "indexedChunks": len(chunks),
        "totalVectors": total_vectors,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "collection": COLLECTION_NAME,
    }

    return {"statusCode": 200, "body": json.dumps(body)}

