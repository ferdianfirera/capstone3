# ingest.py
import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm

# LangChain + Qdrant imports
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load local .env for local dev if present
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def infer_columns(df):
    # best-effort flexible column detection for dataset names
    title_col = next((c for c in ["title","Title","movie","Movie","name","Name"] if c in df.columns), None)
    overview_col = next((c for c in ["overview","plot","Plot","Overview","description","Description"] if c in df.columns), None)
    return title_col, overview_col

def build_documents(df):
    title_col, overview_col = infer_columns(df)
    texts = []
    metadatas = []
    for _, row in df.iterrows():
        title = str(row[title_col]) if title_col else ""
        overview = str(row[overview_col]) if overview_col else ""
        meta = {}
        # capture some common metadata fields if present
        for k in ["year","Year","genre","Genre","director","Director","stars","Stars","imdb_rating","IMDb Rating","rating"]:
            if k in df.columns:
                meta[k.lower()] = row[k]
        meta["title"] = title
        content = f"Title: {title}\n"
        for k,v in meta.items():
            if k != "title":
                content += f"{k}: {v}\n"
        content += "\nOverview:\n" + overview
        texts.append(content)
        metadatas.append(meta)
    return texts, metadatas

def ingest(csv_path, collection_name="movie_database", chunk_size=800, chunk_overlap=100):
    print("Reading CSV:", csv_path)
    df = pd.read_csv(csv_path)
    texts, metadatas = build_documents(df)
    print(f"Built {len(texts)} documents. Splitting into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = []
    doc_metas = []
    for text, meta in tqdm(zip(texts, metadatas), total=len(texts)):
        chunks = text_splitter.split_text(text)
        for i, ch in enumerate(chunks):
            docs.append(ch)
            # keep title + chunk index + small metadata
            doc_metas.append({**meta, "chunk": i, "source": "imdb_top_1000"})

    print("Creating embeddings (OpenAI) and upserting to Qdrant...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

    # This will create the collection and insert vectors
    vectorstore = QdrantVectorStore.from_texts(
        docs,
        embedding=embeddings,
        metadatas=doc_metas,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    print(f"Inserted {len(docs)} chunks into collection `{collection_name}`.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to movies CSV (e.g. data/imdb_top_1000.csv)")
    parser.add_argument("--collection", default="movie_database", help="Qdrant collection name")
    args = parser.parse_args()
    ingest(args.csv, args.collection)
