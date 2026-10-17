
import html
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="BookBuddy | Find Your Next Read",
    page_icon="📚",
    layout="wide",
)


# ---------------- STYLING ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #F7F4ED;
    color: #26352D;
    font-family: 'DM Sans', sans-serif;
}
.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #26352D !important;
}
.hero {
    background: #E6EBDD;
    padding: 35px;
    border-radius: 20px;
    margin-bottom: 25px;
}
.eyebrow {
    color: #39745B;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
}
.book-card {
    background: #FFFFFF;
    padding: 22px;
    border: 1px solid #E5E3DA;
    border-radius: 16px;
    margin-bottom: 14px;
}
.book-title {
    font-family: 'Playfair Display', serif;
    font-size: 21px;
    font-weight: 700;
    color: #26352D;
}
.book-meta {
    color: #39745B;
    font-size: 13px;
    margin-top: 5px;
    margin-bottom: 10px;
}
.book-description {
    color: #626A62;
    font-size: 14px;
    line-height: 1.6;
}
div.stButton > button {
    background: #39745B;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1.3rem;
    font-weight: 600;
    width: 100%;
}
div.stButton > button:hover {
    background: #285640;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD BOOKS ----------------
@st.cache_data
def load_books():
    df = pd.read_csv("books.csv")
    df.columns = df.columns.str.strip().str.lower()

    rename_map = {
        "book_title": "title",
        "name": "title",
        "book name": "title",
        "book_author": "author",
        "category": "genre",
        "summary": "description",
    }
    df = df.rename(columns=rename_map)

    if "title" not in df.columns:
        raise ValueError(
            "books.csv must contain a column named 'title'."
        )

    for column in ["author", "genre", "description"]:
        if column not in df.columns:
            df[column] = "Not specified"

    for column in ["title", "author", "genre", "description"]:
        df[column] = (
            df[column]
            .fillna("Not specified")
            .astype(str)
            .str.strip()
        )

    df = df[df["title"].ne("")]
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

    df["features"] = (
        df["title"] + " " +
        df["author"] + " " +
        df["genre"] + " " +
        df["description"]
    )

    return df


try:
    books = load_books()
except Exception as error:
    st.error(
        "Could not read books.csv. Check its column names and CSV format."
    )
    st.code(str(error))
    st.stop()


if books.empty:
    st.error("Your book library is empty. Add books to books.csv.")
    st.stop()


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL BOOK COMPANION</div>
    <h1 style="font-size:44px; margin-bottom:5px;">
        Find your next favourite read.
    </h1>
    <p style="font-size:16px; color:#526258; max-width:650px;">
        Explore books by genre and discover stories that match your interests.
        Your next great read is just a click away.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.metric("Books in library", len(books))

with col2:
    valid_genres = books.loc[
        ~books["genre"].str.lower().isin(["", "not specified"]),
        "genre"
    ].nunique()
    st.metric("Genres available", valid_genres)

st.divider()


# ---------------- GENRE SELECTION ----------------
st.subheader("Let's find your next read")
st.write("Choose a genre, then select a book you already enjoy.")

genres = sorted(
    books.loc[
        ~books["genre"].str.lower().isin(["", "not specified"]),
        "genre"
    ].unique().tolist(),
    key=str.casefold,
)

genre_options = ["All Genres"] + genres

selected_genre = st.selectbox(
    "1. Select a genre",
    genre_options,
)

if selected_genre == "All Genres":
    genre_books = books.copy()
else:
    genre_books = books[
        books["genre"].str.casefold() == selected_genre.casefold()
    ].copy()

if genre_books.empty:
    st.warning("No books are available in this genre.")
    st.stop()


# ---------------- BOOK SELECTION ----------------
book_titles = sorted(
    genre_books["title"].unique().tolist(),
    key=str.casefold,
)

selected_book = st.selectbox(
    "2. Select a book you like",
    book_titles,
)

# How many other books can be recommended?
max_recommendations = min(10, len(books) - 1)

if max_recommendations < 1:
    st.info("Add at least two books to get recommendations.")
    st.stop()

number_of_recommendations = st.selectbox(
    "3. Number of recommendations",
    list(range(1, max_recommendations + 1)),
    index=min(4, max_recommendations - 1),
)

recommendation_type = st.radio(
    "Recommendation preference",
    [
        "Same genre only",
        "Similar books across all genres",
    ],
    horizontal=True,
    help="Choose whether recommendations must stay in the selected genre.",
)

find_button = st.button(
    "Find my next read",
    use_container_width=True,
)


# ---------------- RECOMMENDATIONS ----------------
if find_button:
    # TF-IDF converts book details into numerical features.
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )

    matrix = vectorizer.fit_transform(books["features"])

    # Calculate similarity between books.
    similarity_matrix = cosine_similarity(matrix)

    selected_matches = books.index[
        books["title"] == selected_book
    ].tolist()

    if not selected_matches:
        st.error("Selected book could not be found.")
        st.stop()

    selected_index = selected_matches[0]

    scores = list(
        enumerate(similarity_matrix[selected_index])
    )

    # Remove the selected book from its own recommendations.
    scores = [
        (index, score)
        for index, score in scores
        if index != selected_index
    ]

    # Keep recommendations within the chosen genre if requested.
    if recommendation_type == "Same genre only":
        allowed_indices = set(genre_books.index)

        scores = [
            (index, score)
            for index, score in scores
            if index in allowed_indices
        ]

    scores.sort(key=lambda item: item[1], reverse=True)

    recommendations = scores[:number_of_recommendations]

    st.divider()
    st.subheader("Your recommended reads")
    st.caption(f"Because you liked {selected_book}")

    if not recommendations:
        st.info(
            "No other books are available for this selection. "
            "Choose another genre or select similar books across all genres."
        )
    else:
        for rank, (index, score) in enumerate(recommendations, start=1):
            book = books.iloc[index]

            title = html.escape(str(book["title"]))
            author = html.escape(str(book["author"]))
            genre = html.escape(str(book["genre"]))
            description = html.escape(str(book["description"]))

            st.markdown(
                f"""
                <div class="book-card">
                    <div class="eyebrow">RECOMMENDATION {rank}</div>
                    <div class="book-title">{title}</div>
                    <div class="book-meta">
                        By {author} &nbsp; • &nbsp; {genre}
                    </div>
                    <div class="book-description">{description}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.progress(
                max(0.0, min(1.0, float(score))),
                text=f"Similarity score: {score * 100:.1f}%",
            )


# ---------------- COMPLETE BOOK LIBRARY ----------------
with st.expander("Explore the complete book library"):
    search_text = st.text_input(
        "Search by title, author, or genre",
        placeholder="e.g. Harry Potter, Rowling, Fantasy",
    )

    display_books = books.copy()

    if selected_genre != "All Genres":
        display_books = display_books[
            display_books["genre"].str.casefold()
            == selected_genre.casefold()
        ]

    if search_text.strip():
        pattern = re.escape(search_text.strip())

        searchable_columns = [
            "title",
            "author",
            "genre",
            "description",
        ]

        mask = display_books[searchable_columns].apply(
            lambda column: column.str.contains(
                pattern,
                case=False,
                na=False,
            )
        ).any(axis=1)

        display_books = display_books[mask]

    st.dataframe(
        display_books[
            ["title", "author", "genre", "description"]
        ],
        use_container_width=True,
        hide_index=True,
    )

st.markdown("""
<div style="text-align:center; color:#7A8279; padding:25px 0 5px;">
    Made for readers, by BookBuddy 📚
</div>
""", unsafe_allow_html=True)
