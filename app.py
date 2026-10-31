
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Text Similarity Checker",
    page_icon="🔍",
    layout="centered"
)

# Load model only once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# App title
st.title("🔍 Text Similarity Checker")
st.write("Compare two texts using sentence embeddings and cosine similarity.")

# User inputs
text1 = st.text_area("Enter Text 1:", height=120)
text2 = st.text_area("Enter Text 2:", height=120)

# Compare button
if st.button("Compare Texts", type="primary"):

    if not text1.strip() or not text2.strip():
        st.warning("Please enter both texts.")

    else:
        # Generate embeddings
        text1_embeddings = model.encode(text1)
        text2_embeddings = model.encode(text2)

        # Calculate cosine similarity
        scores = cosine_similarity(
            [text1_embeddings],
            [text2_embeddings]
        )

        score = float(scores[0][0])

        # Display result
        st.subheader("Similarity Result")

        st.metric("Cosine Similarity Score", f"{score:.4f}")

        # Progress bar (0 to 1)
        st.progress(max(0.0, min(1.0, score)))

        # Classify similarity
        if score > 0.8:
            st.success("The two texts are very similar.")

        elif score < 0.4:
            st.error("The two texts are not very similar.")

        else:
            st.info("The two texts are somewhat similar.")

        # Show original texts
        st.subheader("Compared Texts")
        st.write("**Text 1:**", text1)
        st.write("**Text 2:**", text2)
