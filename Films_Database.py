# films_database.py
from importlib.metadata import metadata
import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from qdrant_client import QdrantClient
import json

COLLECTION_NAME = "movie_collection"
NUMBER_OF_MOVIES_LABEL = "Number of Movies"
ALL_GENRES_OPTION = "All Genres"

# CSS Styles
MOVIE_CARD_GRID_CSS = """
<style>
.movie-card-grid {
    background-color: #ffffff;
    border: 1px solid #dee2e6;
    border-radius: 12px;
    padding: 16px;
    margin: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.3s ease-in-out;
    width: 300px;
    height: 580px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    justify-content: space-between;
}
.movie-card-grid:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    border-color: #007bff;
}
.movie-poster {
    width: 200px;
    height: 300px;
    object-fit: cover !important;
    border-radius: 4px;
    margin-bottom: 12px;
}
.movie-title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 0px;
    color: #333;
    line-height: 1.2;
}
.movie-year {
    color: #666;
    font-size: 18px;
    margin-bottom: 12px;
}
.movie-rating {
    display: flex;
    justify-content: space-between;
    width: 100%;
    margin-bottom: 12px;
}
.movie-details {
    font-size: 18px;
    color: #666;
    margin-bottom: 4px;
    text-align: left;
    width: 100%;
}
.wrapper {
    display: flex;
    flex-direction: column;
    margin-bottom: 8px;
}

</style>
"""

MOVIES_GRID_CSS = """
<style>
.movies-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-start;
    gap: 16px;
    margin: 20px 0;
}
</style>
"""

# HTML Templates
def get_movie_card_html(movie_data):
    """Generate HTML for movie card"""
    return f"""
    <div class="movie-card-grid">
        <div class="wrapper" style="align-items: center;"> 
        <img class="movie-poster" src="{movie_data['poster']}" alt="{movie_data['title']}" onerror="this.src='https://via.placeholder.com/200x300?text=No+Image'">
        <div class="movie-title">{movie_data['title']} ({movie_data['year']})</div>
        </div>
        <div class="wrapper" style="align-items: flex-start;width: 100%;"> 
        <div class="movie-rating">
            <span>⭐ {movie_data['rating']}</span>
        </div>
        <div class="movie-details">🎭 {movie_data['genre']}</div>
        <div class="movie-details">👤 {movie_data['director']}</div>
        <div class="movie-details">⏱️ {movie_data['runtime']}</div>
        </div>
    </div>
    """

# Load environment variables
if "QDRANT_URL" in st.secrets:
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
else:
    load_dotenv()
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

def safe_float(val, default=0.0):
    """Safely convert value to float"""
    try:
        if val and val != '' and val != 'nan':
            return float(str(val).replace(',', ''))
        return default
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    """Safely convert value to integer"""
    try:
        if val and val != '' and val != 'nan':
            return int(str(val).replace(',', ''))
        return default
    except (ValueError, TypeError):
        return default

def create_movie_dict(metadata):
    """Create a movie dictionary from metadata"""
    return {
        'Title': metadata.get('title', 'Unknown'),
        'Year': safe_int(metadata.get('released_year', 0)),
        'Certificate': metadata.get('certificate', 'N/A'),
        'Runtime': metadata.get('runtime', 'N/A'),
        'Genre': metadata.get('genre', 'N/A'),
        'IMDB_Rating': safe_float(metadata.get('imdb_rating', 0)),
        'Meta_Score': safe_float(metadata.get('meta_score', 0)),
        'Director': metadata.get('director', 'N/A'),
        'Star1': metadata.get('star1', 'N/A'),
        'Star2': metadata.get('star2', 'N/A'),
        'Star3': metadata.get('star3', 'N/A'),
        'Star4': metadata.get('star4', 'N/A'),
        'Votes': safe_int(metadata.get('no_of_votes', 0)),
        'Gross': safe_float(metadata.get('gross', 0)),
        'Poster_Link': metadata.get('poster_link', '')
    }

def process_qdrant_points(scroll_result):
    """Process Qdrant scroll result to extract unique movies"""
    movies_data = []
    seen_titles = set()
    
    for point in scroll_result[0]:
        payload = point.payload
        if 'metadata' in payload:
            metadata = payload['metadata']
        else:
            metadata = payload
        title = metadata.get('title', 'Unknown')
        
        # Skip duplicates and unknown titles
        if title not in seen_titles and title != 'Unknown':
            seen_titles.add(title)
            movie = create_movie_dict(metadata)  
            movies_data.append(movie)
    
    return movies_data

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_movies_from_qdrant():
    """Fetch all movie data from Qdrant collection"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Get all points from the collection
        scroll_result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=1000,  # dataset size
            with_payload=True
        )
        
        movies_data = process_qdrant_points(scroll_result)
        return pd.DataFrame(movies_data)
    
    except Exception as e:
        st.error(f"Error fetching data from Qdrant: {e}")
        st.error("Please check:")
        st.error("1. QDRANT_URL and QDRANT_API_KEY are set correctly")
        st.error("2. Your Qdrant collection exists and has data")
        st.error("3. Network connection to Qdrant server")
        
        # Show more debugging info
        with st.expander("Debug Information"):
            st.write(f"QDRANT_URL: {QDRANT_URL}")
            st.write(f"QDRANT_API_KEY: {'Set' if QDRANT_API_KEY else 'Not set'}")
            st.write(f"Collection Name: {COLLECTION_NAME}")
            st.write(f"Error Details: {str(e)}")
        
        return pd.DataFrame()

def create_genre_chart(df):
    """Create genre popularity chart"""
    # Split genres and count them
    all_genres = []
    for genres in df['Genre'].dropna():
        if genres != 'N/A':
            genre_list = [g.strip() for g in str(genres).split(',')]
            all_genres.extend(genre_list)
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    fig = px.bar(
        x=genre_counts.index,
        y=genre_counts.values,
        title='Top 15 Movie Genres',
        labels={'x': 'Genre', 'y': NUMBER_OF_MOVIES_LABEL},
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        xaxis_title="Genre",
        yaxis_title=NUMBER_OF_MOVIES_LABEL,
        xaxis_tickangle=-45,
        showlegend=False
    )
    return fig

def create_year_distribution_chart(df):
    """Create movies by year chart"""
    year_counts = df[df['Year'] > 1900]['Year'].value_counts().sort_index()
    
    fig = px.line(
        x=year_counts.index,
        y=year_counts.values,
        title='Movies Released by Year',
        labels={'x': 'Year', 'y': NUMBER_OF_MOVIES_LABEL}
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=NUMBER_OF_MOVIES_LABEL
    )
    return fig

def display_movie_card_grid(movie_row):
    """Display a movie as a grid card with 300px width"""
    st.markdown(MOVIE_CARD_GRID_CSS, unsafe_allow_html=True)
    
    # Prepare movie data for template
    movie_data = {
        'poster': movie_row['Poster_Link'] if movie_row['Poster_Link'] and movie_row['Poster_Link'] != '' else 'https://via.placeholder.com/200x300?text=No+Image',
        'title': movie_row['Title'],
        'year': movie_row['Year'],
        'rating': movie_row['IMDB_Rating'],
        'meta_score': movie_row['Meta_Score'] if movie_row['Meta_Score'] > 0 else 'N/A',
        'genre': str(movie_row['Genre'])[:25] + ('...' if len(str(movie_row['Genre'])) > 25 else ''),
        'director': str(movie_row['Director'])[:20] + ('...' if len(str(movie_row['Director'])) > 20 else ''),
        'runtime': movie_row['Runtime'],
        'certificate': movie_row['Certificate']
    }
    
    # Generate and display HTML
    card_html = get_movie_card_html(movie_data)
    st.markdown(card_html, unsafe_allow_html=True)

def show_films_database_page():
    """Main function to display the Films Database page"""
    # Page configuration is already set in Home.py, so we skip it here
    st.title("🎬 Films Database & Analytics")

    # Load data
    with st.spinner("Loading movie data..."):
        movies_df = fetch_movies_from_qdrant()

    # Sidebar filters
    st.sidebar.header("Filters")

    # Year filter with robust error handling
    valid_years = movies_df['Year']
    year_min = int(valid_years.min()) if len(valid_years) > 0 else 1920
    year_max = int(valid_years.max()) if len(valid_years) > 0 else 2024

    # Ensure min < max
    if year_min >= year_max:
        year_min = 1920
        year_max = 2024
        st.sidebar.warning("⚠️ Invalid year range detected. Using default range 1900-2024.")

    if year_max > 1900 and year_min < year_max:
        year_range = st.sidebar.slider(
            "Select Year Range",
            year_min,
            year_max,
            (year_min, year_max)
        )
        movies_df = movies_df[(movies_df['Year'] >= year_range[0]) & (movies_df['Year'] <= year_range[1])]

        # Rating filter with robust error handling
        valid_ratings = movies_df['IMDB_Rating']
        rating_min = float(valid_ratings.min()) if len(valid_ratings) > 0 and valid_ratings.min() > 0 else 0.0
        rating_max = float(valid_ratings.max()) if len(valid_ratings) > 0 and valid_ratings.max() > 0 else 10.0

        # Ensure min < max
        if rating_min >= rating_max:
            rating_min = 0.0
            rating_max = 10.0
            st.sidebar.warning("⚠️ Invalid rating range detected. Using default range 0-10.")

        rating_range = st.sidebar.slider(
            "Select IMDB Rating Range",
            rating_min,
            rating_max,
            (rating_min, rating_max),
            step=0.1
        )

        # Apply rating filter
        if len(valid_ratings) > 0 and valid_ratings.max() > 0:
            movies_df = movies_df[(movies_df['IMDB_Rating'] >= rating_range[0]) & (movies_df['IMDB_Rating'] <= rating_range[1])]

        # Genre filter
        st.sidebar.subheader("Genre Filter:")
        all_genres = set()
        for genres in movies_df['Genre'].dropna():
            if genres != 'N/A':
                genre_list = [g.strip() for g in str(genres).split(',')]
                all_genres.update(genre_list)

        # Genre multiselect with "All Genres" option
        selected_genres = st.sidebar.multiselect(
            "Select Genres to Include:",
            sorted(all_genres),
            default=[],
            help="Select 'All Genres' to show all movies, or choose specific genres. Movies matching ANY of the selected genres will be shown."
        )

        # Apply genre filter
        if ALL_GENRES_OPTION not in selected_genres and selected_genres:
            # Specific genres selected - apply filter
            movies_df = movies_df[movies_df['Genre'].apply(
                lambda x: any(genre in str(x) for genre in selected_genres) if pd.notna(x) else False
            )]

        # Reset filters button
        if st.sidebar.button("Reset All Filters", use_container_width=True):
            st.rerun()

    # Charts section
    st.subheader("📊 Analytics Dashboard")

    # Chart selection
    chart_options = {
        "Genre Popularity": "genre_pop",
        "Movies by Year": "year_dist",
    }

    selected_charts = st.multiselect(
        "Select charts to display:",
        options=list(chart_options.keys()),
        default=[]
    )

    # Display selected charts
    for chart_name in selected_charts:
        if chart_options[chart_name] == "genre_pop":
            st.plotly_chart(create_genre_chart(movies_df), use_container_width=True)
        elif chart_options[chart_name] == "year_dist":
            st.plotly_chart(create_year_distribution_chart(movies_df), use_container_width=True)

    # Data table
    st.subheader("💾 Movies Database")

    # Search functionality
    search_term = st.text_input("Search movies by title, director, or actor:")
    if search_term:
        movies_df_before_search = movies_df.copy()
        mask = (
            movies_df['Title'].str.contains(search_term, case=False, na=False) |
            movies_df['Director'].str.contains(search_term, case=False, na=False) |
            movies_df['Star1'].str.contains(search_term, case=False, na=False) |
            movies_df['Star2'].str.contains(search_term, case=False, na=False) |
            movies_df['Star3'].str.contains(search_term, case=False, na=False) |
            movies_df['Star4'].str.contains(search_term, case=False, na=False)
        )
        movies_df = movies_df[mask]
        
        # Show search filter
        st.write(f"**Search results for '{search_term}':** {len(movies_df)} movies found")

    # Sort options
    sort_column = st.selectbox(
        "Sort by:",
        ["Title", "Year", "IMDB_Rating", "Meta_Score", "Votes", "Director"],
        index=2
    )
    sort_order = st.radio("Sort order:", ["Descending", "Ascending"], horizontal=True)
    ascending = sort_order == "Ascending"

    # Sort the dataframe
    movies_df = movies_df.sort_values(by=sort_column, ascending=ascending)

    # Display movies in grid layout
    st.write(f"**Showing {len(movies_df)} movies:**")
    
    # Pagination
    movies_per_page = st.selectbox("Movies per page:", [12, 24, 48], index=1)
    total_pages = (len(movies_df) - 1) // movies_per_page + 1 if len(movies_df) > 0 else 1
    
    if total_pages > 1:
        page_number = st.number_input("Page:", min_value=1, max_value=total_pages, value=1)
    else:
        page_number = 1
    
    # Calculate pagination
    start_idx = (page_number - 1) * movies_per_page
    end_idx = start_idx + movies_per_page
    page_movies = movies_df.iloc[start_idx:end_idx]
    
    # Display pagination info
    if total_pages > 1:
        st.info(f"📄 Page {page_number} of {total_pages} | Showing movies {start_idx + 1}-{min(end_idx, len(movies_df))} of {len(movies_df)}")
    
    # Display movies in grid layout (3 columns)
    movies_list = list(page_movies.iterrows())
    
    # Apply grid CSS
    st.markdown(MOVIES_GRID_CSS, unsafe_allow_html=True)
    
    # Display movies in rows of 3
    num_columns = 4
    for row_start in range(0, len(movies_list), num_columns):
        cols = st.columns(num_columns)
        for col_idx in range(num_columns):
            movie_idx = row_start + col_idx
            if movie_idx < len(movies_list):
                _, movie = movies_list[movie_idx]
                with cols[col_idx]:
                    display_movie_card_grid(movie)

# Main execution (when Films_Database.py is run directly)
if __name__ == "__main__":
    # Page configuration
    st.set_page_config(page_title="Films Database & Analytics", layout="wide")
    show_films_database_page()