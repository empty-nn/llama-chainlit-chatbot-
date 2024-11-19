import os
import openai
from pdfminer.high_level import extract_text
from llama_index.core import Settings, StorageContext
import nest_asyncio
from llama_index.core.tools import QueryEngineTool, ToolMetadata
nest_asyncio.apply()
from llama_index.core import Settings, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# File path for the single PDF book
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

# Step 3: Create an OpenAIAgent using the Book Query Tool
agent = OpenAIAgent.from_tools([query_engine_tool], verbose=True)

print("Book index has been loaded, and agent is ready for queries.")

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = agent.chat(text_input)
    print(f"Agent: {response}")