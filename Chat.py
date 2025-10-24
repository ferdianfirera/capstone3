# moviemaster.py - Main Chat Interface
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Secrets local via .env and Streamlit Cloud via st.secrets)
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

collection_name = "movie_collection"

# Initialize components 
llm = None
embeddings = None
qdrant = None
retriever = None
custom_prompt = None

def initialize_chat_components():
    """Initialize chat components and return success status"""
    global llm, embeddings, qdrant, retriever, custom_prompt
    
    if not OPENAI_API_KEY:
        return False, "OPENAI_API_KEY not set. For local runs put it in a .env file; on Streamlit put it in the app secrets."
    
    try:
        # LLM + Embeddings
        llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.4)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

        # connect to existing Qdrant collection
        qdrant = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        
        retriever = qdrant.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.6})

        # Custom prompt template
        custom_prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=(
                "You are Movie Master, a friendly movie expert.\n"
                "Use the conversation history to maintain context and refer to previously mentioned movies.\n"
                "If asked about a movie, describe it.\n"
                "If asked for recommendations, list 3–5 titles and gives its rating.\n"
                "If the user refers to 'the movies' or 'those movies', use the conversation history to understand which movies they mean from the exact movies mentioned, don't add any new titles.\n"
                "Be concise and engaging.\n\n"
                "CONVERSATION HISTORY:\n{chat_history}\n\n"
                "CONTEXT FROM DATABASE:\n{context}\n\n"
                "CURRENT QUESTION: {question}\n\n"
                "Answer in English:"
            )
        )
        
        return True, "Success"
        
    except Exception as e:
        error_msg = f"Could not connect to Qdrant collection. Make sure you ran ingestion and that QDRANT_URL/QDRANT_API_KEY are correct.\nError: {str(e)}"
        return False, error_msg

def show_chat_page():
    """Main function to display the Chat page"""
    st.title("💬 AI Movie Chat Assistant")
    st.markdown("### Ask me anything about movies!")
    
    success, error_msg = initialize_chat_components()
    
    if not success:
        st.error(error_msg)
        st.markdown("""
        **Requirements:**
        - Set up environment variables (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
        - Ensure your Qdrant collection exists and has data
        - Check network connection to Qdrant server
        """)
        
        # Show manual setup instructions
        with st.expander("🔧 Setup Instructions"):
            st.code("""
            # Set up environment variables in .env file
            OPENAI_API_KEY=your_openai_api_key_here
            QDRANT_URL=your_qdrant_url_here
            QDRANT_API_KEY=your_qdrant_api_key_here
            """)
        return

    # Memory
    if "chat_memory" not in st.session_state:
        st.session_state.chat_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

    # QA Chain
    if "chat_qa_chain" not in st.session_state:
        st.session_state.chat_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=st.session_state.chat_memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] 

    st.markdown("""
    **What you can ask:**
    - Get personalized movie recommendations
    - Ask about specific movies, actors, or directors
    - Find movies based on mood, genre, or theme
    - Get detailed plot summaries and reviews
    - Compare different movies
    """)

    # Sidebar 
    with st.sidebar:
        
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_memory.clear()
            st.session_state.chat_qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                memory=st.session_state.chat_memory,
                return_source_documents=True,
                output_key="answer",
                combine_docs_chain_kwargs={"prompt": custom_prompt}
            )
            st.rerun()
        
        st.divider()
        st.markdown("""
        **💡 Tips:**\n\n
        • Be specific in your questions\n
        • Ask follow-up questions\n
        • Mention genres you like\n
        • Ask about 'those movies' to refer to previous recommendations                    
        """)

        st.divider()
        
        # Debug
        if st.checkbox("🔍 Show conversation memory"):
            if hasattr(st.session_state, 'chat_memory') and st.session_state.chat_memory.chat_memory.messages:
                st.write("**Memory contents:**")
                for msg in st.session_state.chat_memory.chat_memory.messages:
                    st.markdown(f"- {type(msg).__name__}: {msg.content[:100]}...")
            else:
                st.write("Memory is empty")

    # Display past messages
    for role, content in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(content)

    # Chat input
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    if prompt := st.chat_input("Ask me a movie question", key=f"chatinput{st.session_state.input_key}"):
        # append user message to history and UI
        st.session_state.chat_history.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run RAG chain
                    result = st.session_state.chat_qa_chain.invoke({
                        "question": prompt
                    })
                    answer = result.get("answer", "I'm sorry, I couldn't generate a response.")

                    st.session_state.chat_history.append(("assistant", answer))

                    # Display assistant answer
                    st.markdown(answer)
                    st.session_state.input_key += 1
                    st.rerun()

                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append(("assistant", error_msg))
                    st.session_state.input_key += 1
                    st.rerun()

                sources = result.get("source_documents", [])
                
                if sources:
                    st.info("Source is from the retrieval documents.")
                else:
                    st.info("No relevant source documents found.")


# Main execution Chat.py
if __name__ == "__main__":
    st.set_page_config(
        page_title="MovieMaster — IMDB RAG Chat", 
        layout="wide",
        page_icon="🎬"
    )
    show_chat_page()