import streamlit as st
import numpy as np
import librosa
import librosa.display
import sounddevice as sd
import scipy.signal
import tempfile
import os
import pandas as pd
from datetime import datetime
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import matplotlib.pyplot as plt
import soundfile as sf
import subprocess
import warnings
import google.generativeai as genai  # Gemini SDK

warnings.filterwarnings('ignore')

# === DARK THEME ===
def set_dark_theme():
    st.markdown("""
    <style>
    .stApp { background-color: #121212; color: white; }
    .stButton>button { background-color: #3a7ca5; color: white; border-radius: 8px; }
    .stDataFrame { background-color: #252525 !important; }
    </style>
    """, unsafe_allow_html=True)
set_dark_theme()

# === Initialize Gemini API ===
genai.configure(api_key="AIzaSyBSqcuH7a6pSZzjEihtFM1F2gwfhbcuUVw")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# === Gemini Description ===
def get_bird_description_from_gemini(bird_name):
    try:
        prompt = (
            f"Give a short and accurate description of the bird '{bird_name}', "
            f"including its key characteristics and the regions of the world where it is commonly found. "
            f"Provide detailed information about its distribution and habitat."
        )
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
        return "Description unavailable due to Gemini API error."

# === Audio Cleanup ===
def process_audio(y, sr=44100):
    try:
        D = librosa.stft(y)
        magnitude = np.abs(D)
        phase = np.angle(D)
        noise_threshold = np.percentile(magnitude, 20, axis=1, keepdims=True)
        mask = magnitude > (noise_threshold * 1.5)
        y_clean = librosa.istft(magnitude * mask * np.exp(1j * phase))
        sos = scipy.signal.butter(6, [300/(sr/2), 8000/(sr/2)], btype='band', output='sos')
        return librosa.util.normalize(scipy.signal.sosfiltfilt(sos, y_clean))
    except Exception as e:
        st.error(f"Audio processing error: {e}")
        return y

# === SOX Repair ===
def fix_audio_with_sox(original_path):
    try:
        fixed_path = original_path.replace(".wav", "_fixed.wav")
        subprocess.run(['sox', original_path, '-c', '1', '-r', '44100', fixed_path], check=True)
        return fixed_path
    except Exception as e:
        st.error(f"SOX audio repair failed: {e}")
        return None

# === Visualization ===
def create_all_visualizations(y, sr, species=None):
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), facecolor='#121212')

    time = np.linspace(0, len(y)/sr, len(y))
    axs[0].plot(time, y, color='cyan')
    axs[0].set_title('Waveform', color='white')

    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=axs[1], cmap='magma')
    axs[1].set_title('Spectrogram', color='white')

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), y_axis='mel', x_axis='time', sr=sr, ax=axs[2], cmap='plasma')
    axs[2].set_title('Mel Spectrogram', color='white')

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=axs[3], cmap='cool')
    axs[3].set_title('Chromagram', color='white')

    if species:
        fig.suptitle(f"Prediction: {species}", color='yellow', fontsize=16)

    plt.tight_layout()
    return fig

# === MAIN APP ===
def main():
    st.title("🦜 Bird Sound Identifier with Gemini Descriptions")
    min_conf = st.slider("Confidence Threshold (%)", 5, 100, 20) / 100
    duration = st.slider("Recording Duration (sec)", 1, 20, 5)

    if st.button("🎤 Start Recording"):
        with st.spinner(f"Recording for {duration} seconds..."):
            try:
                recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
                sd.wait()
                y = recording.flatten()
                y_processed = process_audio(y)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    sf.write(tmp.name, y_processed, 44100)
                    audio_path = tmp.name

                try:
                    y_lib, sr_lib = sf.read(audio_path)
                    if y_lib.ndim > 1:
                        y_lib = y_lib.mean(axis=1)
                except:
                    st.warning("Audio failed to load. Attempting SOX repair...")
                    audio_path = fix_audio_with_sox(audio_path)
                    y_lib, sr_lib = sf.read(audio_path)

                analyzer = Analyzer()
                rec = Recording(analyzer=analyzer, path=audio_path, date=datetime.now(), min_conf=min_conf, overlap=1.0)
                rec.analyze()

                results = []
                for det in rec.detections:
                    if 'common_name' in det and det['confidence'] >= min_conf:
                        description = get_bird_description_from_gemini(det['common_name'])
                        results.append({
                            "Species": det['common_name'],
                            "Scientific Name": det['scientific_name'],
                            "Confidence": f"{det['confidence']:.1%}",
                            "Description & Region": description,
                            "Method": "Standard Detection"
                        })

                st.audio(audio_path, format='audio/wav')

                if results:
                    st.success(f"Detected: **{results[0]['Species']}**")
                    st.pyplot(create_all_visualizations(y_lib, sr_lib, results[0]['Species']))
                    st.dataframe(pd.DataFrame(results))
                else:
                    st.error("No bird sound detected. Try recording again.")

                os.unlink(audio_path)

            except Exception as e:
                st.error(f"System Error: {e}")

if __name__ == "__main__":
    main()
