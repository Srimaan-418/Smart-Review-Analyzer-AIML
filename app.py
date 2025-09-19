import streamlit as st
import torch
import joblib
import pandas as pd
import requests
from streamlit_lottie import st_lottie
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Smart Review Analyzer",
    page_icon="🧠",
    layout="wide"
)

# --- HELPER FUNCTIONS ---

@st.cache_data
def load_lottieurl(url: str):
    """Loads a Lottie animation from a URL."""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_resource
def load_model_and_artifacts():
    """Loads all necessary model files and artifacts."""
    model = DistilBertForSequenceClassification.from_pretrained('./final_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained('./final_model')
    mlb = joblib.load('mlb.pkl')
    with open('emotions.txt', 'r') as f:
        emotion_names = [line.strip() for line in f.readlines()]
    return model, tokenizer, mlb, emotion_names

def prioritize_review(probabilities, names_list):
    """Applies rules to flag reviews based on detected emotions."""
    try:
        anger_index = names_list.index('anger')
        annoyance_index = names_list.index('annoyance')
        sadness_index = names_list.index('sadness')
    except ValueError:
        return "N/A"
    
    anger_prob = probabilities[anger_index]
    annoyance_prob = probabilities[annoyance_index]
    sadness_prob = probabilities[sadness_index]

    if anger_prob > 0.2 or annoyance_prob > 0.2:
        return "High Priority 🚨"
    elif sadness_prob > 0.2:
        return "Medium Priority ⚠️"
    else:
        return "Low Priority ✅"

def generate_response_template(priority, emotions):
    """Generates a suggested response."""
    if priority == 'High Priority 🚨':
        return "We are very sorry to hear about your negative experience. Your feedback is important, and we want to make things right. Please contact our support team at support@example.com with your user details so we can investigate this for you immediately."
    elif priority == 'Medium Priority ⚠️':
        return "We're sorry that your experience didn't meet your expectations. We appreciate you taking the time to share your feedback and our team will look into the issues you've mentioned."
    elif any(e in emotions for e in ['gratitude', 'joy', 'love', 'admiration']):
        return "Thank you so much for your positive feedback! We're thrilled to hear you had a great experience and appreciate you taking the time to share."
    else:
        return "Thank you for your feedback. We appreciate you sharing your thoughts with us."

# --- LOAD EVERYTHING ---
model, tokenizer, mlb, emotion_names = load_model_and_artifacts()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
lottie_url = "https://assets3.lottiefiles.com/packages/lf20_d2jD1d.json" # Data analysis robot
lottie_animation = load_lottieurl(lottie_url)

# --- USER INTERFACE ---

# Sidebar
with st.sidebar:
    st.title("About the Project")
    st.markdown("This app is a proof-of-concept for the **Smart Review Analyzer** project.")
    st.markdown("It uses a fine-tuned DistilBERT model to perform multi-label emotion classification, prioritizes feedback, and suggests responses.")
    st.markdown("---")
    st.markdown("**By:**")
    st.markdown("Karishma Suthar (2420030384)")
    st.markdown("Bhavya Sai Sri (2420030594)")
    st.markdown("K.Srimaan Kameshwar (2420030418)")

# Main Page Layout with Columns
col1, col2 = st.columns([1, 2])

# Left Column (Animation)
with col1:
    if lottie_animation:
        st_lottie(lottie_animation, height=300, key="homepage_robot")

# Right Column (Main Content)
with col2:
    st.title("✨ Smart Review Analyzer")
    st.markdown("Enter a user review to automatically analyze its emotions, determine its priority, and get a suggested response.")
    user_input = st.text_area(
        "User Review", 
        "This is the best product I have ever used! Highly recommended.", 
        height=150,
        label_visibility="collapsed"
    )

analyze_button = st.button("Analyze Review")

# Results Section
if analyze_button:
    if user_input:
        with st.spinner('Analyzing...'):
            # (Your analysis and results code)
            st.success("Analysis complete!")
