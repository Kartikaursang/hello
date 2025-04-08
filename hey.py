import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import speech_recognition as sr
from st_audiorec import st_audiorec
import numpy as np
import random
import nltk
import io
import wave

# Setup
nltk.download('vader_lexicon', quiet=True)

st.set_page_config(page_title="EmotiCare - Emotion Based AI Chatbot", 
                   page_icon="🎙️", 
                   layout="centered")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert_emotion_model")
model = AutoModelForSequenceClassification.from_pretrained("bert_emotion_model")
model.eval()

emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

responses = {
    "anger": ["Take a deep breath. It's okay to feel this way."],
    "fear": ["You’re safe here. Want to talk about it?"],
    "joy": ["That's amazing! 😊 Keep smiling!"],
    "love": ["Love is powerful. Spread it! 💖"],
    "sadness": ["It's okay to feel sad. I'm here for you."],
    "surprise": ["Oh wow! That’s unexpected!"]
}

video_links = {
    "anger": "https://youtu.be/66gH1xmXkzI",
    "fear": "https://youtu.be/AETFvQonfV8",
    "joy": "https://youtu.be/OcmcptbsvzQ",
    "love": "https://youtu.be/UAaWoz9wJ_4",
    "sadness": "https://youtu.be/W937gFzsD-c",
    "surprise": "https://youtu.be/PE2GkSgOZMA"
}

# ------------------- Functions -------------------

def predict_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits.numpy()[0])
    top_index = np.argmax(probs)
    return emotion_labels[top_index]

def speech_to_text(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_data) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return None

def response_for_emotion(emotion):
    return random.choice(responses.get(emotion, ["I'm here to listen!"]))

# ------------------- UI -------------------

st.title("🎙️ EmotiCare - AI Emotion Detection Chatbot")
st.write("Detect your emotion from **text or voice**, and get emotional support with an AI-powered chatbot!")

input_type = st.radio("Select input method:", ["Text", "Voice"])

if input_type == "Text":
    user_input = st.text_input("💬 Enter your message")
    if st.button("Analyze Emotion"):
        if user_input:
            emotion = predict_emotion(user_input)
            st.markdown(f"### 🎭 Emotion: **{emotion.capitalize()}**")
            st.success(f"🤖 {response_for_emotion(emotion)}")
            st.video(video_links[emotion])
        else:
            st.warning("Please enter a message.")

elif input_type == "Voice":
    st.markdown("### 🎤 Record your voice")
    audio_data = st_audiorec()

    if audio_data is not None:
        # Save the audio bytes
        with wave.open("audio.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(audio_data)

        st.success("Audio Recorded Successfully!")

        with st.spinner("Transcribing..."):
            try:
                text = speech_to_text("audio.wav")
                if text:
                    st.markdown(f"**📝 Transcribed Text:** _{text}_")
                    emotion = predict_emotion(text)
                    st.markdown(f"### 🎭 Emotion: **{emotion.capitalize()}**")
                    st.success(f"🤖 {response_for_emotion(emotion)}")
                    st.video(video_links[emotion])
                else:
                    st.warning("Couldn't detect speech. Try again.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

# ------------------- Sidebar -------------------

st.sidebar.markdown("## 🔍 Features")
st.sidebar.write("- Text/Voice emotion detection")
st.sidebar.write("- AI chatbot replies based on emotion")
st.sidebar.write("- 🎥 Recommended videos for mental wellness")

