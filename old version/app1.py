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
from pathlib import Path
from uuid import uuid4
from collections import deque
import datetime
from typing import Optional, Dict
from llama_index.agent.openai import OpenAIAgent
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

os.environ["OAUTH_GOOGLE_CLIENT_ID"] = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
os.environ["OAUTH_GOOGLE_CLIENT_SECRET"] = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

nest_asyncio.apply()

# Specify the years
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
storage_dir = "./storage/book"  # Make sure this matches the directory you used for storing the book index

# Step 1: Load the Storage Context and Book Index
storage_context = StorageContext.from_defaults(
    persist_dir=storage_dir
)

# Load the previously saved index for the book using load_index_from_storage
book_index = load_index_from_storage(storage_context)

# Step 2: Create a Query Engine Tool for the Book Index
query_engine_tool = QueryEngineTool(
    query_engine=book_index.as_query_engine(),
    metadata=ToolMetadata(
        name="book_index",
        description="Index for the book - useful for answering questions about the basic economics"
    )
)

# Create message history for chat session
message_history = deque(maxlen=10)  # Keep the last 10 messages
# Create tools for each year's index if not already created

agent = OpenAIAgent.from_tools([query_engine_tool], verbose=True)
@cl.on_chat_start
async def start():
    # Limit message history to last 10 messages
    cl.user_session.set("message_history", message_history)

    # Send the initial message
    await cl.Message(
        author="Assistant",
        content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

async def set_sources(response, msg):
    elements = []
    label_list = []
    for count, sr in enumerate(response.source_nodes, start=1):
        elements.append(cl.Text(
            name="S" + str(count),
            content=f"{sr.node.text}",
            display="side",
            size="small",
        ))
        label_list.append("S" + str(count))
    msg.elements = elements
    await msg.update()

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history", deque(maxlen=10))
    
    if message_history is None:
        message_history = deque(maxlen=10)
        cl.user_session.set("message_history", message_history)
    
    msg = cl.Message(content="", author="Assistant")
    user_message = message.content
    
    # Process the user's query asynchronously
    res = agent.chat(message.content)  # Ensure this is awaited if it's an async function
    
    # await cl.Message(content=res).send()
    # Check if 'res' has a 'response' attribute for the full response-w
    if res and hasattr(res, 'response'):
        msg.content = res.response
        message_history.append({"author": "user", "content": user_message})
        message_history.append({"author": "Assistant", "content": msg.content})
        message_history = list(message_history)[-4:]  # Keep the last 4 messages
        cl.user_session.set("message_history", message_history)
    else:
        msg.content = "I couldn't process your query. Please try again."
    await msg.send()
    
    if res and hasattr(res, 'source_nodes'):
        await set_sources(res, msg)

@cl.on_chat_resume
async def resume():

    await cl.Message(content="Welcome back! How can I assist you today?").send()

@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User
) -> Optional[cl.User]:
    """Handle Google OAuth callback."""
    print("OAuth callback received from provider:", provider_id)
    print("Token:", token)
    print("Raw user data:", raw_user_data)

    # Check if the provider is Google and process user information
    if provider_id == "google":
        user_email = raw_user_data.get("email")
        if user_email:
            return cl.User(identifier=user_email, metadata={"role": "user"})
    return None