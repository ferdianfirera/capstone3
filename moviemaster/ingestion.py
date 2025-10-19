# import basics
import os
from dotenv import load_dotenv

# import qdrant
from qdrant_client import QdrantClient, models

# import langchain components
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings

# documents
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --- QDRANT CONFIGURATION ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "movie_database2") # Default to 'movie_database'
DIMENSION = 3072 # The dimension for 'text-embedding-3-large'

# Initialize Qdrant Client
print("Initializing Qdrant Client...")
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


# Check whether collection exists, and create if not
try:
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' already exists. Skipping creation.")
except Exception:
    print(f"Collection '{COLLECTION_NAME}' does not exist. Creating...")
    
    # Use recreate_collection to ensure it starts fresh or is created correctly
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=DIMENSION, distance=models.Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created with dimension {DIMENSION}.")


# initialize embeddings model + vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Initialize QdrantVectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    # This setting is crucial for the Qdrant client to perform well
    # Setting parallel to 2 to use 2 threads for concurrent upsert operations
    # This helps speed up the ingestion process
    batch_size=100,
)


# loading the document
loader = CSVLoader("documents/imdb_top_1000_movies.csv")

print("Loading raw documents...")
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} raw documents.")

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
print("Splitting documents into chunks...")
documents = text_splitter.split_documents(raw_documents)
print(f"Created {len(documents)} chunks for ingestion.")

# generate unique id's (using the provided method)
i = 0
uuids = []

while i < len(documents):
    i += 1
    uuids.append(f"id{i}")

# add to database
print(f"Adding {len(documents)} chunks to Qdrant collection '{COLLECTION_NAME}'...")
vector_store.add_documents(documents=documents, ids=uuids)

print("Ingestion complete!")
