
import csv
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="BookBuddy | Book Recommendations",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- DARK THEME ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #080a09;
    color: #f4f5f4;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: #080a09;
}

[data-testid="stSidebar"] {
    background: #111512;
    border-right: 1px solid #252b26;
}

[data-testid="stSidebar"] * {
    color: #f4f5f4 !important;
}

h1, h2, h3, h4, p, label, li {
    color: #f4f5f4;
}

h1, h2 {
    font-family: 'Playfair Display', serif;
}

.hero {
    background: linear-gradient(135deg, #17271d, #101411 70%);
    border: 1px solid #2c4232;
    border-radius: 22px;
    padding: 35px 30px;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin: 0 0 8px 0;
    color: #ffffff;
}

.hero p {
    color: #c2cec4;
    font-size: 16px;
    margin: 0;
}

.eyebrow {
    color: #78d69a;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.metric-card {
    background: #141915;
    border: 1px solid #29342b;
    padding: 20px;
    border-radius: 15px;
    min-height: 100px;
}

.metric-label {
    color: #a8b5aa;
    font-size: 13px;
    margin-bottom: 8px;
}

.metric-value {
    color: #85e3a4;
    font-size: 27px;
    font-weight: 700;
}

.book-card {
    background: #131714;
    border: 1px solid #2b332d;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    min-height: 150px;
}

.book-title {
    color: #ffffff;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 7px;
}

.book-author {
    color: #9fcea9;
    font-size: 13px;
    margin-bottom: 12px;
}

.book-description {
    color: #c4cbc5;
    font-size: 13px;
    line-height: 1.6;
}

.genre-tag {
    display: inline-block;
    color: #a4edb9;
    background: #203b29;
    border: 1px solid #31583b;
    border-radius: 20px;
    padding: 4px 10px;
    font-size: 11px;
    margin-top: 12px;
}

div[data-baseweb="select"] > div {
    background-color: #191e1a;
    color: #ffffff;
    border-color: #39453b;
}

div[data-baseweb="select"] span {
    color: #ffffff;
}

input, textarea {
    background-color: #191e1a !important;
    color: #ffffff !important;
    border-color: #39453b !important;
}

.stButton > button {
    background: #75d995;
    color: #071009;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    min-height: 42px;
}

.stButton > button:hover {
    background: #9aefb2;
    color: #071009;
    border: none;
}

[data-testid="stExpander"] {
    background: #121613;
    border: 1px solid #2b332d;
    border-radius: 10px;
}

[data-testid="stMetric"] {
    background: #141915;
    border: 1px solid #29342b;
    border-radius: 12px;
    padding: 14px;
}

[data-testid="stMetricValue"] {
    color: #85e3a4;
}

[data-testid="stMetricLabel"] {
    color: #b5c0b7;
}

hr {
    border-color: #29342b;
}

.stCaption {
    color: #aab5ac;
}

footer {
    visibility: hidden;
}

#MainMenu {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ---------------- LOAD BOOK DATA ----------------

@st.cache_data
def load_books():
    file_path = Path(__file__).parent / "books.csv"

    if not file_path.exists():
        return pd.DataFrame(
            columns=["title", "author", "genre", "description"]
        ), f"Could not find books.csv at {file_path}"

    # Read CSV safely, including descriptions that contain commas.
    rows = []

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file)
            all_rows = list(reader)

        if not all_rows:
            return pd.DataFrame(
                columns=["title", "author", "genre", "description"]
            ), "Your books.csv file is empty."

        headers = [
            str(column).strip().lower().replace(" ", "_")
            for column in all_rows[0]
        ]

        header_map = {
            "book_title": "title",
            "name": "title",
            "book_name": "title",
            "writer": "author",
            "book_author": "author",
            "category": "genre",
            "book_genre": "genre",
            "summary": "description",
            "details": "description",
            "book_description": "description",
        }

        headers = [header_map.get(h, h) for h in headers]

        # Support CSV files with or without a book ID column.
        has_id = any(h in headers for h in ["book_id", "id", "isbn"])

        for row in all_rows[1:]:
            if not row or not any(str(x).strip() for x in row):
                continue

            if has_id:
                if len(row) < 5:
                    continue

                # Join extra columns back into the description.
                values = row[:4] + [",".join(row[4:])]
                record = dict(zip(headers[:4] + ["description"], values))

                # If the original CSV has five columns, use its actual headers.
                record = {
                    headers[i]: row[i] for i in range(min(4, len(headers), len(row)))
                }
                record["description"] = ",".join(row[4:])

                # Remove the ID field from the final book data.
                record.pop("book_id", None)
                record.pop("id", None)
                record.pop("isbn", None)

            else:
                if len(row) < 4:
                    continue

                record = {
                    headers[0]: row[0],
                    headers[1]: row[1],
                    headers[2]: row[2],
                    headers[3]: ",".join(row[3:]),
                }

            rows.append(record)

        books = pd.DataFrame(rows)

        for column in ["title", "author", "genre", "description"]:
            if column not in books.columns:
                books[column] = ""

            books[column] = (
                books[column]
                .fillna("")
                .astype(str)
                .str.strip()
            )

        books = books[
            (books["title"] != "") &
            (books["genre"] != "")
        ].copy()

        books = books.drop_duplicates(
            subset=["title", "author"],
            keep="first"
        ).reset_index(drop=True)

        if books.empty:
            return books, (
                "No valid books were found. Check that books.csv has "
                "title, author, genre and description columns."
            )

        return books, None

    except Exception as error:
        return pd.DataFrame(
            columns=["title", "author", "genre", "description"]
        ), f"Could not read books.csv: {error}"


books, load_error = load_books()


# ---------------- RECOMMENDATION ENGINE ----------------

def recommend_books(selected_title, count, same_genre):
    if books.empty:
        return pd.DataFrame()

    book_data = books.copy()

    book_data["combined"] = (
        book_data["title"] + " " +
        book_data["author"] + " " +
        book_data["genre"] + " " +
        book_data["description"]
    )

    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000
        )

        matrix = vectorizer.fit_transform(book_data["combined"])
        similarity = cosine_similarity(matrix)

        matching_indices = book_data.index[
            book_data["title"] == selected_title
        ].tolist()

        if not matching_indices:
            return pd.DataFrame()

        selected_index = matching_indices[0]
        scores = list(enumerate(similarity[selected_index]))

        scores.sort(key=lambda item: item[1], reverse=True)

        selected_genre = book_data.loc[selected_index, "genre"]
        results = []

        for index, score in scores:
            if index == selected_index:
                continue

            if same_genre and book_data.loc[index, "genre"] != selected_genre:
                continue

            results.append(index)

            if len(results) >= count:
                break

        return book_data.loc[results].drop(
            columns=["combined"],
            errors="ignore"
        )

    except Exception as error:
        st.error(f"Recommendation error: {error}")
        return pd.DataFrame()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown(
        "<div class='eyebrow'>YOUR READING SPACE</div>",
        unsafe_allow_html=True
    )
    st.markdown("## 📚 BookBuddy")
    st.caption("Discover your next favourite book.")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Discover Books", "Book Library", "About"],
        index=0
    )

    st.divider()

    st.markdown("### Library status")

    st.markdown(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>Books available</div>"
        f"<div class='metric-value'>{len(books)}</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    if not books.empty:
        st.caption(f"{books['genre'].nunique()} genres to explore")


# ---------------- HERO HEADER ----------------

st.markdown("""
<div class="hero">
    <div class="eyebrow">A LITTLE SOMETHING FOR YOUR NEXT CHAPTER</div>
    <h1>Find your next favourite book.</h1>
    <p>Explore stories, discover new genres, and get recommendations made from book details.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- ERROR HANDLING ----------------

if load_error:
    st.error(load_error)
    st.info(
        "Check that books.csv is in the same GitHub folder as "
        "streamlit_app.py and that its header contains "
        "book_id, title, author, genre, description."
    )
    st.stop()


# ---------------- DISCOVER BOOKS ----------------

if page == "Discover Books":

    # Summary cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Total books</div>"
            f"<div class='metric-value'>{len(books)}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Genres</div>"
            f"<div class='metric-value'>{books['genre'].nunique()}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>Authors</div>"
            f"<div class='metric-value'>{books['author'].nunique()}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("## Discover books")
    st.write("Choose a genre and a book to find similar reads.")

    genres = sorted(books["genre"].dropna().unique().tolist())

    genre_options = ["All Genres"] + genres

    selected_genre = st.selectbox(
        "Choose a genre",
        genre_options,
        index=0
    )

    if selected_genre == "All Genres":
        filtered_books = books.copy()
    else:
        filtered_books = books[
            books["genre"] == selected_genre
        ].copy()

    if filtered_books.empty:
        st.warning("No books are available in this genre.")
        st.stop()

    book_options = sorted(filtered_books["title"].unique().tolist())

    selected_title = st.selectbox(
        "Choose a book you like",
        book_options
    )

    selected_book = books[
        books["title"] == selected_title
    ].iloc[0]

    with st.expander("View selected book details", expanded=True):
        st.markdown(
            f"<div class='book-title'>{html.escape(selected_book['title'])}</div>"
            f"<div class='book-author'>By {html.escape(selected_book['author'])}</div>"
            f"<div class='book-description'>{html.escape(selected_book['description'])}</div>"
            f"<span class='genre-tag'>{html.escape(selected_book['genre'])}</span>",
            unsafe_allow_html=True
        )

    st.markdown("### Recommendation settings")

    setting1, setting2 = st.columns(2)

    with setting1:
        recommendation_count = st.selectbox(
            "Number of recommendations",
            [3, 5, 8, 10],
            index=1
        )

    with setting2:
        recommendation_mode = st.selectbox(
            "Recommendation type",
            [
                "Similar books across all genres",
                "Same genre only"
            ]
        )

    if st.button("✨ Find my next read", use_container_width=True):
        same_genre_only = recommendation_mode == "Same genre only"

        recommendations = recommend_books(
            selected_title,
            recommendation_count,
            same_genre_only
        )

        if recommendations.empty:
            st.info(
                "No similar books were found. Try another book or select "
                "'Similar books across all genres'."
            )
        else:
            st.session_state["bookbuddy_recommendations"] = recommendations
            st.session_state["bookbuddy_selected"] = selected_title

    if "bookbuddy_recommendations" in st.session_state:
        st.markdown("## Your recommended reads")
        st.caption(
            f"Because you selected "
            f"{st.session_state['bookbuddy_selected']}."
        )

        recommendations = st.session_state["bookbuddy_recommendations"]

        card_columns = st.columns(2)

        for index, (_, book) in enumerate(recommendations.iterrows()):
            with card_columns[index % 2]:
                st.markdown(
                    f"<div class='book-card'>"
                    f"<div class='book-title'>{html.escape(str(book['title']))}</div>"
                    f"<div class='book-author'>By {html.escape(str(book['author']))}</div>"
                    f"<div class='book-description'>{html.escape(str(book['description']))}</div>"
                    f"<span class='genre-tag'>{html.escape(str(book['genre']))}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )


# ---------------- BOOK LIBRARY ----------------

elif page == "Book Library":

    st.markdown("## Explore the library")
    st.write("Search the collection or filter it by genre.")

    search_text = st.text_input(
        "Search books, authors or descriptions",
        placeholder="Try Harry Potter, fantasy, habits..."
    )

    genre_filter = st.selectbox(
        "Filter by genre",
        ["All Genres"] + sorted(books["genre"].unique().tolist())
    )

    library = books.copy()

    if genre_filter != "All Genres":
        library = library[library["genre"] == genre_filter]

    if search_text.strip():
        query = search_text.strip().lower()

        mask = (
            library["title"].str.lower().str.contains(query, regex=False) |
            library["author"].str.lower().str.contains(query, regex=False) |
            library["genre"].str.lower().str.contains(query, regex=False) |
            library["description"].str.lower().str.contains(query, regex=False)
        )

        library = library[mask]

    st.caption(f"Showing {len(library)} books")

    if library.empty:
        st.info("No books matched your search. Try a different keyword.")
    else:
        for start in range(0, len(library), 2):
            columns = st.columns(2)

            for offset, column in enumerate(columns):
                position = start + offset

                if position >= len(library):
                    break

                book = library.iloc[position]

                with column:
                    st.markdown(
                        f"<div class='book-card'>"
                        f"<div class='book-title'>{html.escape(str(book['title']))}</div>"
                        f"<div class='book-author'>By {html.escape(str(book['author']))}</div>"
                        f"<div class='book-description'>{html.escape(str(book['description']))}</div>"
                        f"<span class='genre-tag'>{html.escape(str(book['genre']))}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )


# ---------------- ABOUT ----------------

elif page == "About":

    st.markdown("## About BookBuddy")

    st.markdown("""
    BookBuddy is a simple content-based book recommendation system.

    ### How does it work?

    1. **Select a book:** Choose a title from the available library.
    2. **Process book details:** The application uses the title, author,
       genre and description.
    3. **TF-IDF:** Text is converted into numerical features.
    4. **Cosine similarity:** The application compares the feature vectors
       to find books with similar content.
    5. **Get recommendations:** The most similar books are displayed.

    ### Technologies used

    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - TF-IDF and cosine similarity

    ### Project limitation

    Recommendations depend on the book information in the CSV file.
    The system does not learn from personal ratings or reading history.
    """)
    
# ---------------- FOOTER ----------------

st.divider()

st.markdown(
    "<p style='text-align:center;color:#849187;font-size:12px;'>"
    "BOOKBUDDY · A simple space for discovering your next read"
    "</p>",
    unsafe_allow_html=True
)
