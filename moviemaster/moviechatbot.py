# import streamlit
import streamlit as st
import os
from dotenv import load_dotenv

# import langchain
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# import qdrant client
from qdrant_client import QdrantClient

load_dotenv()

st.title("RAG Chatbot (Powered by Qdrant)")

# --- QDRANT CONFIGURATION ---
QDRANT_URL = os.environ.get("QDRANT_URL")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION_NAME", "movie_database2")

# Initialize Qdrant Client
# This client is used to connect to the Qdrant service
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# initialize embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# initialize vector store (connecting to the existing Qdrant collection)
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings
)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Initialize with a basic SystemMessage (not the context-dependent one)
    st.session_state.messages.append(SystemMessage("You are an assistant for question-answering tasks. "))

# display chat messages from history on app rerun
for message in st.session_state.messages:
    # Skip displaying the context-dependent SystemMessage
    if isinstance(message, SystemMessage) and message.content.startswith("You are an assistant for question-answering tasks. Use the following pieces of retrieved context"):
        continue

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# create the bar where we can type messages
prompt = st.chat_input("Ask about the movies in the database...")

# did the user submit a prompt?
if prompt:

    # add the message from the user (prompt) to the screen with streamlit
    with st.chat_message("user"):
        st.markdown(prompt)

    # Note: We append the HumanMessage here, but we DON'T append the SystemMessage 
    # used for context, as that would confuse the chat history display logic above.
    st.session_state.messages.append(HumanMessage(prompt))

    # initialize the llm
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0, # Changed temperature to 0 for more consistent Q&A based on RAG
        api_key=os.environ.get("OPENAI_API_KEY") # Ensure API key is passed here too
    )

    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.7}, # Increased score_threshold for better relevance
    )

    docs = retriever.invoke(prompt)
    docs_text = "\n---\n".join(d.page_content for d in docs)

    # creating the system prompt
    system_prompt = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Context:
    ---
    {context}
    ---"""

    # Populate the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # The messages list for the LLM call should include the context prompt first,
    # followed by the latest human prompt. We will create a temporary list for the LLM.
    llm_messages = [
        SystemMessage(content=system_prompt_fmt),
        HumanMessage(content=prompt)
    ]

    print("-- SYS PROMPT --")
    print(system_prompt_fmt)
    
    # invoking the llm
    result = llm.invoke(llm_messages).content

    # adding the response from the llm to the screen (and chat)
    with st.chat_message("assistant"):
        st.markdown(result)

        st.session_state.messages.append(AIMessage(result))
