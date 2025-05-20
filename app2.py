import pickle
import streamlit as st
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -- API Configuration ------------------------------------------------------
API_KEY           = "8265bd1679663a7ea12ac168da84d2e8"
TMDB_MOVIE_URL    = "https://api.themoviedb.org/3/movie/{}"
TMDB_IMAGE_BASE   = "https://image.tmdb.org/t/p/w500/"

# -- Retry Setup ------------------------------------------------------------
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)

# -- Hybrid Recommendation Core ---------------------------------------------
def hybrid_recommendation(movie_title, user_id=1, alpha=0.6):
    try:
        movie_idx = movies.index[movies['title'] == movie_title][0]
        content_scores = list(enumerate(content_sim[movie_idx]))
        tmdb_ids = movies['movie_id'].tolist()
        collab_scores = [
            pred_df.loc[user_id, m_id] if (user_id in pred_df.index and m_id in pred_df.columns)
            else 2.5
            for m_id in tmdb_ids
        ]
        combined = [
            (i, alpha * c + (1 - alpha) * cf)
            for (i, c), cf in zip(content_scores, collab_scores)
        ]
        top_indices = [i for i, _ in sorted(combined, key=lambda x: x[1], reverse=True)[1:6]]
        return top_indices
    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return []

# -- TMDb Data Fetcher ------------------------------------------------------
def fetch_movie_data(movie_id):
    try:
        resp = session.get(
            TMDB_MOVIE_URL.format(movie_id),
            params={"api_key": API_KEY},
            timeout=10
        )
        data = resp.json()
        poster = data.get("poster_path")
        return (
            data.get("overview", "No description."),
            TMDB_IMAGE_BASE + poster if poster else None
        )
    except Exception as e:
        st.warning(f"Couldn't fetch data: {e}")
        return "No description.", None

# -- Streamlit UI -----------------------------------------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")
st.header("🎬 Hybrid Movie Recommender System")

# Load models
with st.spinner("Loading models..."):
    movies        = pickle.load(open("model/movies.pkl",      "rb"))
    content_sim   = pickle.load(open("model/content_sim.pkl", "rb"))
    collab_models = pickle.load(open("model/collab.pkl",      "rb"))
    pred_df       = collab_models['pred_df']

# UI: Movie select only
selected_movie = st.selectbox("Select a movie you like:", movies['title'])

if st.button("Get Recommendations"):
    with st.spinner("Computing hybrid recommendations…"):
        rec_idxs = hybrid_recommendation(selected_movie)
        if not rec_idxs:
            st.error("No recommendations could be generated.")
            st.stop()

        st.subheader("Recommended Movies")
        for idx in rec_idxs:
            mid       = movies.at[idx, 'movie_id']
            title     = movies.at[idx, 'title']
            overview, poster = fetch_movie_data(mid)
            cols = st.columns([1,3])
            with cols[0]:
                if poster:
                    st.image(poster, width=200)
                else:
                    st.write("No poster available")
            with cols[1]:
                st.markdown(f"### {title}")
                st.write(overview)
            st.markdown("---")
