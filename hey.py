import os
import numpy as np
import torch
import speech_recognition as sr
import nltk
import streamlit as st
import random
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Initialize NLTK
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load BERT sentiment model
MODEL_PATH = "bert_emotion_model"  # Replace with actual model directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Emotion labels (customize based on your model's label order)
bert_emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']
id2label = {i: label for i, label in enumerate(bert_emotion_labels)}

# Predefined emotion responses
emotion_responses = {
    "anger": ["I understand. Take a deep breath. Want some help calming down?", "Try some relaxation techniques!"],
    "fear": ["It's okay to feel scared. Want to talk about it?", "Fear is natural. Take a deep breath."],
    "joy": ["That's wonderful! 😊 Let's keep the good vibes going!", "Joy is a beautiful feeling. Celebrate it!"],
    "love": ["That's heartwarming! Spread the love 💖", "Love makes the world brighter. Tell someone you care!"],
    "sadness": ["I'm here for you. Want to talk about it? 💙", "Things will get better. Stay strong!"],
    "surprise": ["Oh wow! That sounds interesting!", "Surprises can be exciting! Want to share more?"]
}

# Video suggestions
video_links = {
    "anger": "https://youtu.be/66gH1xmXkzI?si=zDv3BIHqQqXYlEqx",
    "fear": "https://youtu.be/AETFvQonfV8?si=h7JWyBwTyYPKwqtc",
    "joy": "https://youtu.be/OcmcptbsvzQ?si=hUQtzH0vRyGV5hmK",
    "love": "https://youtu.be/UAaWoz9wJ_4?si=Qktt7mDUXRmFda5t",
    "sadness": "https://youtu.be/W937gFzsD-c?si=aT3DcRssJRdF0SeH",
    "surprise": "https://youtu.be/PE2GkSgOZMA?si=yZwanX7PC16C73SG"
}

# Convert speech to text
def speech_to_text(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio)
        except (sr.UnknownValueError, sr.RequestError):
            return None

# BERT-based sentiment analysis
def analyze_sentiment_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits[0].numpy()
        probs = softmax(scores)
        predicted_label = np.argmax(probs)
    return id2label[predicted_label]

# Chatbot response generator
def generate_response(emotion):
    return random.choice(emotion_responses.get(emotion, ["I'm here to chat! 😊"]))

# Streamlit UI setup
st.set_page_config(page_title="EmotiCare - Emotion Based AI Chatbot", page_icon="🎙️", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #FFDEE9, #B5FFFC);
        font-family: 'Arial', sans-serif;
    }
    .title {
        color: #ff4b5c;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subtitle {
        color: #333;
        font-size: 18px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ EmotiCare - Emotion Based AI Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions from your voice or text and receive personalized responses!</p>', unsafe_allow_html=True)

st.sidebar.title("🔍 Choose Input Method")
input_type = st.sidebar.radio("", ["Text", "Voice"])

# Text-based emotion detection
if input_type == "Text":
    user_text = st.text_input("💬 Type your message:")
    if st.button("Analyze Emotion 🎭", key="analyze_text"):
        if user_text:
            with st.spinner("🔍 Analyzing Emotion..."):
                detected_emotion = analyze_sentiment_bert(user_text)
            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
            st.success(f"🤖 Chatbot: {generate_response(detected_emotion)}")
            if detected_emotion in video_links:
                st.markdown("### 📺 **Check out this video!**")
                st.video(video_links[detected_emotion])
        else:
            st.warning("⚠️ Please enter some text.")

# Voice-based emotion detection via upload
elif input_type == "Voice":
    uploaded_audio = st.file_uploader("🎙️ Upload your voice (WAV format)", type=["wav"])

    if uploaded_audio is not None:
        with open("user_voice.wav", "wb") as f:
            f.write(uploaded_audio.read())

        if st.button("Analyze Voice Emotion 🎭", key="analyze_voice"):
            with st.spinner("🎧 Processing Audio..."):
                text = speech_to_text("user_voice.wav")
                if text:
                    detected_emotion = analyze_sentiment_bert(text)
                else:
                    detected_emotion = "neutral"

            if text:
                st.markdown(f"### 📝 **Recorded Text:** _{text}_")

            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
            st.success(f"🤖 Chatbot: {generate_response(detected_emotion)}")
            if detected_emotion in video_links:
                st.markdown("### 📺 **Check out this video!**")
                st.video(video_links[detected_emotion])

# Sidebar Info
st.sidebar.markdown("## 🌟 Features")
st.sidebar.write("✅ Detect emotion from **text** or **voice**")  
st.sidebar.write("✅ Get AI-generated **chatbot responses**")  
st.sidebar.write("✅ Receive **video recommendations**")  
st.sidebar.write("✅ Modern UI with smooth gradients")  
