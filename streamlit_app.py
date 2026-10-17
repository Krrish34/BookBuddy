
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="BookBuddy | Discover Your Next Read",
    page_icon="📚",
    layout="wide"
)

# ---------- PREMIUM DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: linear-gradient(135deg, #101827, #17263a 60%, #24364b);
    color: #f7f0e5;
    font-family: 'DM Sans', sans-serif;
}
header[data-testid="stHeader"] {
    background: transparent;
}
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    font-family: 'Playfair Display', Georgia, serif !important;
    color: #fff3df !important;
}
p, label, .stMarkdown {
    color: #e4e2dd;
}
.hero {
    padding: 35px;
    border: 1px solid #435166;
    border-radius: 24px;
    background: linear-gradient(120deg, #26364b, #192538);
    margin-bottom: 25px;
}
.eyebrow {
    color: #e6b96d;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
}
.hero-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(38px, 5vw, 60px);
    line-height: 1.1;
    color: #fff3df;
    margin: 14px 0;
}
.hero-copy {
    font-size: 16px;
    color: #cbd4df;
    max-width: 650px;
}
.stat {
    background: #202f43;
    border: 1px solid #3c4a5e;
    border-radius: 15px;
    padding: 17px;
    text-align: center;
}
.stat-number {
    font-size: 27px;
    font-weight: 700;
    color: #e6b96d;
}
.stat-label {
    font-size: 12px;
    color: #cbd4df;
}
.book-card {
    background: #f6efe3;
    color: #253044;
    padding: 22px;
    border-radius: 17px;
    border: 1px solid #e7d6bb;
    min-height: 180px;
    margin-bottom: 15px;
}
.book-card h3 {
    color: #253044 !important;
    font-size: 21px;
    margin-bottom: 7px;
}
.book-card p {
    color: #5b6573;
    font-size: 14px;
}
.tag {
    display: inline-block;
    background: #ead7b8;
    color: #65461f;
    border-radius: 20px;
    padding: 5px 10px;
    font-size: 11px;
    font-weight: 700;
    margin: 5px 0;
}
.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 28px;
    color: #fff3df;
    margin-top: 20px;
}
div.stButton > button {
    background: #e6b96d;
    color: #182337;
    border: none;
    border-radius: 10px;
    min-height: 45px;
    font-weight: 700;
    transition: 0.2s;
}
div.stButton > button:hover {
    background: #f3d49d;
    color: #101827;
    border: none;
}
div[data-baseweb="select"] > div {
    background: #f6efe3;
    color: #253044;
    border-radius: 10px;
}
div[data-testid="stTextInput"] input {
    background: #f6efe3;
    color: #253044;
    border-radius: 10px;
}
hr {
    border-color: #3c4a5e;
}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD DATA ----------
@st.cache_data
def load_books():
    df = pd.read_csv("books.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

st.markdown("""
<div class="hero">
    <div class="eyebrow">YOUR PERSONAL READING COMPANION</div>
    <div class="hero-title">Every book leads<br>to another world.</div>
    <div class="hero-copy">
        Discover stories, explore new ideas, and find your next favourite
        book with personalised content-based recommendations.
    </div>
</div>
""", unsafe_allow_html=True)

try:
    books = load_books()

    title_col = next(
        (c for c in ["title", "book_title", "name"] if c in books.columns),
        None
    )
    author_col = next(
        (c for c in ["author", "authors", "writer"] if c in books.columns),
        None
    )
    genre_col = next(
        (c for c in ["genre", "genres", "category", "subject"] if c in books.columns),
        None
    )

    if title_col is None:
        st.error("A title column was not found in books.csv.")
        st.write("Your CSV columns:", list(books.columns))
        st.stop()

    books = books.dropna(subset=[title_col]).reset_index(drop=True)
    books[title_col] = books[title_col].astype(str)
    books = books.drop_duplicates(subset=[title_col]).reset_index(drop=True)

    # Build recommendation features from available book information
    feature_cols = [
        c for c in books.columns
        if c != title_col and c.lower() not in ["id", "book_id", "isbn", "rating"]
    ]

    if feature_cols:
        features = books[feature_cols].fillna("").astype(str).agg(" ".join, axis=1)
    else:
        features = books[title_col]

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(features)
    similarity = cosine_similarity(matrix)

    # ---------- QUICK STATS ----------
    a, b, c = st.columns(3)
    with a:
        st.markdown(
            f'<div class="stat"><div class="stat-number">{len(books)}</div>'
            '<div class="stat-label">BOOKS IN LIBRARY</div></div>',
            unsafe_allow_html=True
        )
    with b:
        st.markdown(
            f'<div class="stat"><div class="stat-number">{len(feature_cols)}</div>'
            '<div class="stat-label">BOOK DATA FIELDS</div></div>',
            unsafe_allow_html=True
        )
    with c:
        st.markdown(
            '<div class="stat"><div class="stat-number">AI + ML</div>'
            '<div class="stat-label">RECOMMENDATION ENGINE</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">Find your next read</div>',
                unsafe_allow_html=True)
    st.write("Choose a book you enjoy. BookBuddy will find similar books.")

    titles = books[title_col].tolist()
    selected = st.selectbox("I recently enjoyed...", titles)

    left, right = st.columns([2, 1])
    with left:
        count = st.slider(
            "How many recommendations?",
            1, min(5, max(1, len(books) - 1)),
            min(3, max(1, len(books) - 1))
        )
    with right:
        st.write("")
        st.write("")
        recommend = st.button("✨ Discover Books", use_container_width=True)

    if recommend:
        index = titles.index(selected)
        scores = list(enumerate(similarity[index]))
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [(i, score) for i, score in scores if i != index][:count]

        st.markdown('<div class="section-title">Picked just for you</div>',
                    unsafe_allow_html=True)
        st.caption("Recommendations are ranked by content similarity.")

        columns = st.columns(min(3, max(1, len(results))))
        for position, (i, score) in enumerate(results):
            row = books.iloc[i]
            author = str(row[author_col]) if author_col else "Author not listed"
            genre = str(row[genre_col]) if genre_col else "Discover"
            with columns[position % len(columns)]:
                st.markdown(f"""
                <div class="book-card">
                    <div style="font-size:30px">📖</div>
                    <span class="tag">{genre}</span>
                    <h3>{row[title_col]}</h3>
                    <p>By {author}</p>
                    <p>Similarity score: {score:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

    with st.expander("📚 Explore the book collection"):
        search = st.text_input("Search by book title")
        display = books.copy()
        if search:
            display = display[
                display[title_col].str.contains(search, case=False, na=False)
            ]
        st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#b6c1cf'>"
        "BOOKBUDDY · Read more. Discover more. 📚</p>",
        unsafe_allow_html=True
    )

except FileNotFoundError:
    st.error("Could not find books.csv. Keep it in the same folder as streamlit_app.py.")
except Exception as e:
    st.error(f"Something went wrong: {e}")
