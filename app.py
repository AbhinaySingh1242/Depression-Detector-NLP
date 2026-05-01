import streamlit as st
import pickle
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Depression Detection",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# CLEAN UI STYLING
# -----------------------------
st.markdown("""
<style>

/* Light Background */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1521737604893-d14cc237f11d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Light overlay */
.stApp::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255,255,255,0.75);
    z-index: 0;
}

/* Content above overlay */
.block-container {
    position: relative;
    z-index: 1;
}

/* Title */
h1 {
    color: #1a1a1a;
    text-align: center;
    font-weight: 700;
}

/* Button */
div.stButton > button {
    background-color: #1976d2;
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 10px 24px;
    transition: 0.3s;
}

div.stButton > button:hover {
    background-color: #0d47a1;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# -----------------------------
# CLEAN TEXT FUNCTION
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# -----------------------------
# TITLE
# -----------------------------
st.markdown("<h1>🧠 Depression Detection App</h1>", unsafe_allow_html=True)

st.markdown("### ✍️ Enter your thoughts and check your mental health status")

# -----------------------------
# INPUT
# -----------------------------
user_input = st.text_area("Write your feelings here:", height=150)

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("🔍 Predict"):
    if user_input:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned]).toarray()

        with st.spinner("Analyzing your text..."):
            result = model.predict(vector)

        if result[0] == 1:
            st.error("😔 You may be experiencing signs of depression")
        else:
            st.success("😊 You seem mentally fine")
    else:
        st.warning("⚠️ Please enter some text")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>Made with ❤️ using NLP & Machine Learning</p>", unsafe_allow_html=True)