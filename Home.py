# Import necessary libraries
import streamlit as st
import os
from dotenv import load_dotenv

# Import LangChain components for chat functionality
try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="MovieMaster",
    page_icon="🎬",
    layout="wide"
)

# Import Films Database functionality
try:
    from Films_Database import show_films_database_page
    FILMS_DB_AVAILABLE = True
except ImportError:
    FILMS_DB_AVAILABLE = False
    st.error("Could not import Films_Database.py. Please ensure the file is in the same directory.")

# Import Chat functionality
try:
    from Chat import show_chat_page
    CHAT_AVAILABLE = True
except ImportError:
    CHAT_AVAILABLE = False

# Initialize navigation
if "page" not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.header("🎬 MovieMaster")
    
    # Navigation dropdown
    page_options = {
        "🏠 Home": "home",
        "💬 Chat Assistant": "chat", 
        "📊 Films Database": "database"
    }
    
    # Find current page display name
    current_page_display = next(
        (display_name for display_name, page_value in page_options.items() 
         if page_value == st.session_state.page), 
        "🏠 Home"
    )
    
    selected_page_display = st.selectbox(
        "Select Page:",
        options=list(page_options.keys()),
        index=list(page_options.keys()).index(current_page_display),
        help="Choose a page to navigate to"
    )
    
    # Update session state if selection changed
    selected_page = page_options[selected_page_display]
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

# Route to different pages based on session state
if st.session_state.page == "home":
    # HOME PAGE CONTENT
    st.title("🎬 Welcome to MovieMaster")
    st.markdown("### Your AI-Powered Movie Companion")

    # Hero section
    st.markdown("""
        **MovieMaster** is your intelligent movie assistant powered by advanced AI and a comprehensive IMDB database. 
        Whether you're looking for personalized recommendations, detailed movie information, or data insights, 
        we've got you covered!
        
        ### Features:
        - 🤖 **AI Chat Assistant**: Get personalized movie recommendations and detailed information
        - 📊 **Films Database**: Browse and analyze movies with interactive charts and filters  
        - 🔍 **Smart Search**: Find movies by title, director, actor, genre, and more
        - 📈 **Analytics Dashboard**: Visualize movie trends, ratings, and statistics
        """)

elif st.session_state.page == "chat":
    # CHAT PAGE CONTENT
    if CHAT_AVAILABLE:
        show_chat_page()
    else:
        st.title("💬 AI Movie Chat Assistant")
        st.markdown("### Ask me anything about movies!")
        
        st.error("Chat functionality is not available. Please check the following:")
        st.markdown("""
        **Requirements:**
        - Ensure `Chat.py` file exists in the same directory
        - Install required packages: `pip install langchain langchain-openai langchain-qdrant`
        - Set up environment variables (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
        """)
        
        # Show manual setup instructions
        with st.expander("🔧 Manual Setup Instructions"):
            st.code("""
            # Install required packages
            pip install langchain langchain-openai langchain-qdrant

            # Set up environment variables in .env file
            OPENAI_API_KEY=your_openai_api_key_here
            QDRANT_URL=your_qdrant_url_here
            QDRANT_API_KEY=your_qdrant_api_key_here
            """)
            
        if st.button("🔄 Retry Loading", width="stretch"):
            st.rerun()

elif st.session_state.page == "database":
    # FILMS DATABASE PAGE CONTENT
    if FILMS_DB_AVAILABLE:
        show_films_database_page()
    else:
        st.error("Films Database module is not available.")
        st.markdown("""
        **Error Details:**
        - Could not import `Films_Database.py`
        - Please ensure the file exists in the same directory
        - Check that all required dependencies are installed
        """)
        
        if st.button("🔄 Retry Loading", width="stretch"):
            st.rerun()
