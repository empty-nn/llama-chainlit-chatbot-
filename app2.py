import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)
import openai
from llama_index.agent.openai import OpenAIAgent
from pathlib import Path
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
import nest_asyncio
from pathlib import Path
from typing import Optional, Dict
from chainlit.context import get_context
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.memory import ChatMemoryBuffer
from chainlit.types import ThreadDict

# Set API keys from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OAUTH_GOOGLE_CLIENT_ID"] = os.getenv("OAUTH_GOOGLE_CLIENT_ID")
os.environ["OAUTH_GOOGLE_CLIENT_SECRET"] = os.getenv("OAUTH_GOOGLE_CLIENT_SECRET")

openai.api_key = os.getenv("OPENAI_API_KEY")
nest_asyncio.apply()

Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
storage_dir = "./storage/book"  # Make sure this matches the directory you used for storing the book index

# Step 1: Load the Storage Context and Book Index
storage_context = StorageContext.from_defaults(
    persist_dir=storage_dir
)

book_index = load_index_from_storage(storage_context)

query_engine_tool = QueryEngineTool(
    query_engine=book_index.as_query_engine(),
    metadata=ToolMetadata(
        name="book_index",
        description="Index for the book - useful for answering questions about the basic economics"
    )
)

def initialize_chatbot_for_years(memory):
    # memory = ChatMemoryBuffer.from_defaults(
    #     token_limit=3000,
    #     chat_store=chat_store,
    #     chat_store_key= chatid  # Using username as a key to store conversation history
    # )

    agent = OpenAIAgent.from_tools(
        tools=[query_engine_tool],
        memory=memory
    )

    return agent

@cl.on_chat_start
async def initialize_chat_session():
    await cl.Message(
        author="assistant",
        content="Hello! I'm an AI assistant. How may I help you?"
    ).send()

@cl.on_message
async def handle_user_message(message: cl.Message):
    # Retrieve the chat store path from the user session
    history_path = cl.user_session.get("history_path")
    memory = cl.user_session.get("memory")

    # Initialize context to get thread_id for unique identification
    context = get_context()
    thread_id = context.session.thread_id

    # Determine if this is a new session or an existing one
    if history_path is None:
        # Create a new path for the chat history based on thread_id
        history_path = Path(f"./history/{thread_id}.json")
        # Ensure the directory exists
        history_path.parent.mkdir(parents=True, exist_ok=True)
        # Initialize a new chat store
        chat_store = SimpleChatStore()
        # Save history_path in the session for future use
        cl.user_session.set("history_path", str(history_path))
        memory = ChatMemoryBuffer.from_defaults(
            token_limit=3000,
            chat_store=chat_store,
            chat_store_key= thread_id  # Using username as a key to store conversation history
        )
        cl.user_session.set("memory", memory)
    # else:
    #     # Load the existing chat store from the specified history file
    #     history_path = Path(history_path)  # Convert to Path object if it was stored as a string
    #     chat_store = SimpleChatStore.from_persist_path(str(history_path))

    # Extract the content of the user's message
    message_content = message.content

    # Initialize an agent (assuming this uses chat_store)
    agent = initialize_chatbot_for_years(memory)

    # Generate a response from the assistant
    response = str(agent.chat(message_content))

    # Persist the updated chat store to the history file
    memory.chat_store.persist(str(history_path))
    
    # Send the assistant's response back to the user
    await cl.Message(content=response).send()


@cl.on_chat_resume
async def resume_chat_session(thread: ThreadDict):
    history_path = cl.user_session.get("history_path")
    history_path = Path(history_path)  # Convert to Path object if it was stored as a string

    # if history_path is None:
    #     # Create a new path for the chat history based on thread_id
    #     history_path = Path(f"./history/{thread_id}.json")
    #     # Ensure the directory exists
    #     history_path.parent.mkdir(parents=True, exist_ok=True)
    #     # Initialize a new chat store
    #     chat_store = SimpleChatStore()
    #     # Save history_path in the session for future use
    #     cl.user_session.set("history_path", str(history_path))
    #     memory = ChatMemoryBuffer.from_defaults(
    #         token_limit=3000,
    #         chat_store=chat_store,
    #         chat_store_key= thread_id  # Using username as a key to store conversation history
    #     )

    chat_store = SimpleChatStore.from_persist_path(str(history_path))
    
    context = get_context()
    thread_id = context.session.thread_id

    memory = ChatMemoryBuffer.from_defaults(
        token_limit=3000,
        chat_store=chat_store,
        chat_store_key= thread_id  # Using username as a key to store conversation history
    )
    cl.user_session.set("memory", memory)
    
# Step 6: OAuth Callback Handling
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
