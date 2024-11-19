import os
import openai
from pdfminer.high_level import extract_text
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core import Document
from pathlib import Path

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ["OPENAI_API_KEY"]
# File path for the single PDF book
book_path = Path("./data/economics.pdf")  # Adjust the path to your PDF file as needed

# Function to extract text from PDFs using PDFMiner
def read_pdf(file_path):
    try:
        text = extract_text(file_path)
        return text
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""

# Extract the PDF content using PDFMiner
pdf_content = read_pdf(book_path)

# Create document entry and add metadata
book_docs = [{"content": pdf_content, "metadata": {"title": "Basic Economics"}}]

# Convert documents into the required format for indexing
documents = [Document(text=doc["content"], metadata=doc["metadata"]) for doc in book_docs]

# Define chunk size for indexing
Settings.chunk_size = 512

# Create a storage directory for the book index
storage_dir = Path("./storage/book")
storage_dir.mkdir(parents=True, exist_ok=True)

# Create a storage context and vector store index
storage_context = StorageContext.from_defaults()
book_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

# Persist the storage context to the storage directory
storage_context.persist(persist_dir=storage_dir)

print("Index for the book has been created and saved.")
