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

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set. Add it to a .env file or set the environment variable.")
if not QDRANT_URL or not QDRANT_API_KEY:
    print("Warning: QDRANT_URL or QDRANT_API_KEY not set. You can still prepare documents locally, but upsert to Qdrant will fail until keys are set.")

def infer_columns(df: pd.DataFrame):
    title_col = next((c for c in ["title","Title","movie","Movie","name","Name"] if c in df.columns), None)
    overview_col = next((c for c in ["overview","plot","Plot","Overview","description","Description","storyline"] if c in df.columns), None)
    return title_col, overview_col

def build_documents(df: pd.DataFrame):
    title_col, overview_col = infer_columns(df)
    if not title_col and not overview_col:
        raise ValueError("Couldn't find a title or overview column in the CSV. Columns found: %s" % list(df.columns))
    texts = []
    metadatas = []
    for _, row in df.iterrows():
        title = str(row[title_col]) if title_col and pd.notna(row[title_col]) else "Unknown Title"
        overview = str(row[overview_col]) if overview_col and pd.notna(row[overview_col]) else ""

        # Build a stable metadata dict with a title key
        meta = {"title": title}
        # collect extra known fields if present
        for k in ["year","Year","genre","Genre","director","Director","stars","Stars","rating","imdb_rating","imdb"]:
            if k in df.columns and pd.notna(row[k]):
                meta_key = k.lower()
                meta[meta_key] = row[k]

        content = f"Title: {title}\n"
        # append metadata lines
        for mk, mv in meta.items():
            if mk != "title":
                content += f"{mk}: {mv}\n"
        content += "\nOverview:\n" + overview


        texts.append(content)
        metadatas.append(meta)


    return texts, metadatas


def ingest(csv_path: str, collection_name: str = "movie_master", chunk_size: int = 800, chunk_overlap: int = 100):
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
            doc_metas.append({**meta, "chunk": i, "source": "imdb_top_1000"})

    print("Creating embeddings (OpenAI) and upserting to Qdrant...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY) 

    # If Qdrant credentials are not set the following will fail; handled by user configuration
    if not QDRANT_URL or not QDRANT_API_KEY:
        # fallback: just create embeddings locally and save to disk for debugging
        try:
            import json
            out = []
            for i, d in enumerate(docs):
                out.append({"id": i, "text": d, "meta": doc_metas[i]})
            with open("local_chunks_preview.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
            print("Qdrant credentials missing. Saved chunk preview to local_chunks_preview.json. You can set QDRANT_URL and QDRANT_API_KEY to upsert to Qdrant.")
            return
        except Exception as e:
            raise RuntimeError("Qdrant credentials missing and failed to write local preview: %s" % e)           

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
    parser.add_argument("--csv", required=True, help="Path to movies CSV (e.g. data/movies.csv)")
    parser.add_argument("--collection", default="movie_master", help="Qdrant collection name")
    args = parser.parse_args()
    ingest(args.csv, args.collection)    
