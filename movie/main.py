# moviemaster.py
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# ---- secrets (works both locally via .env and on Streamlit Cloud via st.secrets) ----
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- LLM + Embeddings ----
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# ---- connect to existing Qdrant collection ----
collection_name = "movie_database"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

retriever = qdrant.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20})

# Custom prompt template
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are MovieMaster, an expert movie recommender that speaks English only.\n"
        "Your task: read the provided movie summaries and recommend or explain movies "
        "that best match the user's question.\n\n"
        "Use only the information in the context below. Never answer in another language.\n"
        "If the user asks for recommendations, suggest 2–3 movies from the context and explain briefly why.\n"
        "If the context doesn't contain relevant movies, say you couldn't find suitable matches.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer in English:"
    )
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)


# ---- Streamlit UI ----
st.set_page_config(page_title="MovieMaster — IMDB RAG Chat", layout="wide")
st.title("🎬 MovieMaster — ask the IMDB Top 1000")

if "history" not in st.session_state:
    st.session_state.history = []  # list of (user, bot) tuples

with st.sidebar:
    st.header("Instructions")
    st.write(
        "Ask about any movie, director, actor, synopsis, or ask for suggestions.\n\n"
        "Make sure you've run `ingest.py` to populate the Qdrant collection first."
    )
    if st.button("Clear chat"):
        st.session_state.history = []

# input area
query = st.text_input("Ask a question about the IMDB Top 1000 movies")

if st.button("Send") and query:
    with st.spinner("Searching Qdrant + calling LLM..."):
        # supply chat_history as list of tuples for the chain
        chat_history = st.session_state.history.copy()
        # LangChain expects a list of tuples or messages; we use (user,bot) tuples
        result = qa_chain({"question": query, "chat_history": chat_history})
        # read answer and sources
        answer = result.get("answer") or result.get("output_text") or result.get("result") or ""
        sources = result.get("source_documents", [])

        # store history
        st.session_state.history.append((query, answer))

        # show bot answer
        st.markdown("**Answer:**")
        st.success(answer)

        # show sources (title/year + snippet)
        if sources:
            st.markdown("**Sources (retrieved chunks):**")
            for doc in sources:
                meta = doc.metadata or {}
                title = meta.get("title") or meta.get("name") or meta.get("movie") or "Unknown"
                year = meta.get("year", "")
                snippet = (doc.page_content or "")[:400].strip().replace("\n", " ")
                st.write(f"- **{title}** {year} — {snippet}")
