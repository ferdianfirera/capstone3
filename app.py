import streamlit as st
from Films_Database import show_films_database_page
from Chat import show_chat_page

# Set page configuration
st.set_page_config(
    page_title="MovieMaster",
    page_icon="🎬",
    layout="wide"
)

PAGES = {
    "🏠 Home": "home",
    "💬 Chat Assistant": "chat",
    "📊 Films Database": "database"
}

# Sidebar navigation
st.sidebar.header("🎬 MovieMaster")
choice = st.sidebar.selectbox("Navigate to:", list(PAGES.keys()))
selected_page = PAGES[choice]

# Route to different pages
if selected_page == "home":
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

elif selected_page == "chat":
    # CHAT PAGE CONTENT
    try:
        show_chat_page()
    except Exception as e:
        st.title("💬 AI Movie Chat Assistant")
        st.error(f"Chat functionality error: {e}")
        st.markdown("""
        **Requirements:**
        - Ensure all required packages are installed
        - Set up environment variables (OPENAI_API_KEY, QDRANT_URL, QDRANT_API_KEY)
        """)

elif selected_page == "database":
    # FILMS DATABASE PAGE CONTENT
    try:
        show_films_database_page()
    except Exception as e:
        st.title("📊 Films Database & Analytics")
        st.error(f"Database functionality error: {e}")
        st.markdown("""
        **Error Details:**
        - Could not load Films Database module
        - Please ensure all required dependencies are installed
        """)
