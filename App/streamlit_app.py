import streamlit as st
import pickle

# Custom page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Custom CSS styling for larger font and spacing
st.markdown("""
    <style>
        .main {
            font-family: 'Arial';
        }
        h1 {
            font-size: 3em !important;
            color: #2c3e50;
            text-align: center;
        }
        .stTextArea textarea {
            font-size: 1.2em;
            height: 200px;
        }
        .stButton button {
            font-size: 1.5em;
            background-color: #2c3e50;
            color: white;
            border-radius: 8px;
            padding: 0.75em 1.5em;
        }
        .stAlert {
            font-size: 1.3em;
        }
    </style>
""", unsafe_allow_html=True)

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Title
st.title("📰 Fake News Detector")

st.markdown("### 📝 Paste a news article or sentence below:")

# Input field
user_input = st.text_area("", "")

# Prediction
if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        vec_input = vectorizer.transform([user_input])
        result = model.predict(vec_input)[0]
        if result == 1:
            st.success("✅ This seems to be **REAL** news.")
        else:
            st.error("🚨 This seems to be **FAKE** news.")
