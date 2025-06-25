
import streamlit as st
import librosa
import numpy as np
import soundfile as sf
import pickle
import os

# Load the trained model
@st.cache_resource
def load_model():
    with open("emotion_model.pkl", "rb") as file:
        return pickle.load(file)

# Load label encoder
@st.cache_resource
def load_label_encoder():
    with open("label_encoder.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
label_encoder = load_label_encoder()

# Extract full 179 features (same as training)
def extract_features(file_path):
    try:
        y_audio, sr = librosa.load(file_path, sr=22050)

        n_mfcc = 40
        n_mels = 125
        n_chroma = 10

        mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)

        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel)
        mel_mean = np.mean(mel_db, axis=1)

        zcr = np.mean(librosa.feature.zero_crossing_rate(y_audio))
        bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y_audio, sr=sr))
        chroma = librosa.feature.chroma_stft(y=y_audio, sr=sr, n_chroma=n_chroma)
        chroma_mean = np.mean(chroma, axis=1)

        # Default values for intensity and gender during inference
        features = np.concatenate([
            mfcc_mean, mel_mean, [zcr], [bandwidth], chroma_mean, [1], [1]
        ])

        return features

    except Exception as e:
        st.error(f"Error during feature extraction: {e}")
        return None

# Streamlit app layout
st.set_page_config(page_title="Emotion Detection from Audio", layout="centered")
st.title("🎧 Emotion Detection from Voice")
st.markdown("Upload a `.wav` audio file to detect the **emotion** expressed in it.")

uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

if uploaded_file is not None:
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(uploaded_file, format='audio/wav')

    features = extract_features(temp_path)

    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        predicted_index = np.argmax(prediction)
        predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]
        st.success(f"🎯 Predicted Emotion: **{predicted_emotion.capitalize()}**")

    if os.path.exists(temp_path):
        os.remove(temp_path)
