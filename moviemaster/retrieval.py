# import basics
import os
from dotenv import load_dotenv

# import qdrant components
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore

# import langchain
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

# --- QDRANT CONFIGURATION ---
# Ensure QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY are set in your .env file
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "movie_database2") # Use a default collection name

# Initialize Qdrant Client
print("Initializing Qdrant Client...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# initialize vector store (connecting to the existing Qdrant collection)
# The vector store will connect to the collection specified by COLLECTION_NAME
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# retrieval
print(f"Retrieving documents from collection: {COLLECTION_NAME}")
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 5, "score_threshold": 0.5},
)
results = retriever.invoke("what is movie about space exploration?")

# show results
print("--- RETRIEVAL RESULTS ---")

for i, res in enumerate(results):
    # Print only the first 100 characters of the content for brevity
    content_preview = res.page_content[:100].replace('\n', ' ') + '...'
    print(f"Result {i+1}:")
    print(f"  Content: {content_preview}")
    print(f"  Metadata: {res.metadata}")

print("Retrieval test complete.")
