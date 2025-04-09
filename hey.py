import streamlit as st
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import librosa
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

# Load BERT model and tokenizer
BERT_MODEL_PATH = "bert_emotion_model"

try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    st.success("✅ BERT emotion model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load BERT emotion model: {e}")

# Emotion labels
bert_emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

# Function to convert audio to text using speech recognition
def speech_to_text_from_audio(audio_data):
    recognizer = sr.Recognizer()
    with sr.AudioData(audio_data.tobytes(), 16000, 2) as source_audio:
        try:
            text = recognizer.recognize_google(source_audio)
            return text
        except sr.UnknownValueError:
            return None

# Predict emotion from text using BERT model
def predict_bert_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = softmax(outputs.logits.numpy()[0])
    top_index = np.argmax(scores)
    return bert_emotion_labels[top_index]

# Record audio using the microphone
def record_audio(duration=5, samplerate=16000):
    """ Record audio using sounddevice library and specify input device """
    devices = sd.query_devices()
    st.write(devices)  # Display all available devices in the Streamlit UI
    
    # Check if there are any input devices
    if not any(dev['max_input_channels'] > 0 for dev in devices):
        st.error("⚠️ No audio input devices found. Please check your microphone.")
        return None

    # Select the first input device
    input_device = None
    for dev in devices:
        if dev['max_input_channels'] > 0:
            input_device = dev
            break

    if input_device is None:
        st.error("⚠️ No microphone found in the available devices.")
        return None
    
    st.write(f"Using device: {input_device['name']}")
    
    # Record audio for the specified duration
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    return audio_data

# Web UI with Streamlit
st.title("🎙️ EmotiCare - Emotion Based AI Chatbot")
st.write("Record your voice to detect emotions or type a message!")

input_type = st.sidebar.radio("Choose Input Method", ["Text", "Voice"])

if input_type == "Text":
    user_text = st.text_input("💬 Type your message:")
    if st.button("Analyze Emotion 🎭"):
        if user_text:
            with st.spinner("🔍 Analyzing Emotion..."):
                detected_emotion = predict_bert_emotion(user_text)
            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
        else:
            st.warning("⚠️ Please enter some text.")

elif input_type == "Voice":
    st.write("🎙️ Recording your voice...")

    # Record audio using the microphone
    audio_data = record_audio(duration=5)  # Record for 5 seconds
    if audio_data is not None:
        # Process the recorded audio
        st.write("🎧 Processing audio...")
        text = speech_to_text_from_audio(audio_data)
        if text:
            st.markdown(f"### 📝 **Transcribed Text:** _{text}_")
            detected_emotion = predict_bert_emotion(text)
            st.markdown(f"### 🎭 Detected Emotion: **{detected_emotion.capitalize()}**")
        else:
            st.warning("⚠️ Could not transcribe the audio. Please try again.")
