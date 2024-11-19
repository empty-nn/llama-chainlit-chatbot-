import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import openai
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from collections import deque
from literalai import LiteralClient
import nest_asyncio

from uuid import uuid4
from collections import deque
import datetime
from typing import Optional, Dict
from llama_index.agent.openai import OpenAIAgent

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

os.environ["LITERAL_API_KEY"] = os.getenv("LITERAL_API_KEY")
lai = LiteralClient(api_key=os.environ.get("LITERAL_API_KEY"))
os.environ["OAUTH_GOOGLE_CLIENT_ID"] = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
os.environ["OAUTH_GOOGLE_CLIENT_SECRET"] = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

nest_asyncio.apply()

# Specify years and storage paths
years = [2019, 2020, 2021, 2022, 2023]
loader = UnstructuredReader()
doc_set = {}
all_docs = []
for year in years:
    year_docs = loader.load_data(
        file=Path(f"./data/MICROSOFT/10-K-{year}.html"), split_documents=False
    )
    for d in year_docs:
        d.metadata = {"year": year}
    doc_set[year] = year_docs
    all_docs.extend(year_docs)

Settings.chunk_size = 512
index_set = {}

for year in years:
    storage_dir = Path(f"./storage/{year}")
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(doc_set[year], storage_context=storage_context)
    index_set[year] = cur_index
    storage_context.persist(persist_dir=storage_dir)