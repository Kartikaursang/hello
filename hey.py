import os
import librosa
import numpy as np
import random
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import speech_recognition as sr
import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# Set page config
st.set_page_config(page_title="EmotiCare - Emotion Based AI Chatbot", 
                   page_icon="🎙️", 
                   layout="centered")

# Download NLTK resources for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

# Load the BERT model for emotion prediction
BERT_MODEL_PATH = "bert_emotion_model"
try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    st.success("✅ BERT emotion model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load BERT emotion model: {e}")

# Emotion labels (customize as per your model training order)
bert_emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# Define responses based on detected emotions
emotion_responses = {
    "anger": ["I understand. Take a deep breath. Want some help calming down?", "Try some relaxation techniques!"],
    "fear": ["It's okay to feel scared. Want to talk about it?", "Fear is natural. Take a deep breath."],
    "joy": ["That's wonderful! 😊 Let's keep the good vibes going!", "Joy is a beautiful feeling. Celebrate it!"],
    "love": ["That's heartwarming! Spread the love 💖", "Love makes the world brighter. Tell someone you care!"],
    "sadness": ["I'm here for you. Want to talk about it? 💙", "Things will get better. Stay strong!"],
    "surprise": ["Oh wow! That sounds interesting!", "Surprises can be exciting! Want to share more?"]
}

# Function to convert speech to text
def speech_to_text_from_audio(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioData(audio_data.tobytes(), 16000, 2) as source_audio:
        try:
            text = recognizer.recognize_google(source_audio)
            return text
        except sr.UnknownValueError:
            return None

# Function to predict emotion using the BERT model
def predict_bert_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    top_index = np.argmax(scores)
    return bert_emotion_labels[top_index]

# Function to generate responses based on emotion
def generate_response(emotion):
    return random.choice(emotion_responses.get(emotion, ["I'm here to chat! 😊"]))

# WebRTC Audio Processor class (Only handles audio)
class AudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        return np.array(frame.to_ndarray())  # Return the raw audio frame as ndarray

# ---------------- UI ----------------
st.markdown("""
    <style>
    .stApp { background: linear-gradient(to right, #FFDEE9, #B5FFFC); font-family: 'Arial', sans-serif; }
    .title { color: #ff4b5c; font-size: 36px; font-weight: bold; text-align: center; }
    .subtitle { color: #333; font-size: 18px; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="title">🎙️ EmotiCare - Emotion Based AI Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect emotions from your voice and receive personalized responses!</p>', unsafe_allow_html=True)

st.sidebar.title("🔍 Choose Input Method")
input_type = st.sidebar.radio("", ["Voice"])

# Voice input processing
if input_type == "Voice":
    st.write("🎙️ Record your voice:")
    
    # Set up the WebRTC audio streamer for receiving audio
    webrtc_ctx = webrtc_streamer(
        key="audio-input-demo",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # When WebRTC starts playing
    if webrtc_ctx.state.playing:
        if webrtc_ctx.audio_receiver:
            audio_data = webrtc_ctx.audio_receiver.get_audio()  # Retrieve audio from WebRTC receiver
            if audio_data is not None:
                st.write("🎧 Processing audio...")
                text = speech_to_text_from_audio(audio_data)  # Convert audio to text
                if text:
                    st.markdown(f"### 📝 **Transcribed Text:** _{text}_")
                    detected_emotion = predict_bert_emotion(text)  # Predict emotion based on text
                    st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
                    st.success(f"🤖 Chatbot: {generate_response(detected_emotion)}")  # Generate response based on emotion
                else:
                    st.warning("⚠️ Could not detect any speech. Please try again.")
        else:
            st.warning("⚠️ Waiting for audio...")

# ---------------- Sidebar Features ----------------
st.sidebar.markdown("## 🌟 Features")
st.sidebar.write("✅ Detect emotion from voice input")  
st.sidebar.write("🤖 AI-generated chatbot responses")  
st.sidebar.write("📺 Video recommendations based on emotion (if needed)")
