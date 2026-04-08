"""
Bird Sound Analysis Script (b.py)
==================================
Standalone analysis module for bird audio processing.
Uses Groq LLM for species description generation and includes
deep audio feature extraction capabilities.

This module can be used independently or imported by the backend.
"""

import numpy as np
import librosa
import librosa.display
import scipy.signal
import tempfile
import os
import pandas as pd
from datetime import datetime
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import subprocess
import warnings
from groq import Groq
from dotenv import load_dotenv

from model.feature_extractor import AudioFeatureExtractor

warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

# === Initialize Groq API ===
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "llama-3.3-70b-versatile"

# === Initialize Feature Extractor ===
feature_extractor = AudioFeatureExtractor()


def get_bird_description_from_groq(bird_name):
    """Generate bird species description using Groq LLM (llama-3.3-70b-versatile)."""
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ornithologist. Provide concise, accurate, "
                        "and informative descriptions of bird species."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Give a short and accurate description of the bird '{bird_name}', "
                        f"including its key characteristics and the regions of the world "
                        f"where it is commonly found. Provide detailed information about "
                        f"its distribution and habitat."
                    ),
                },
            ],
            model=GROQ_MODEL,
            temperature=0.5,
            max_completion_tokens=500,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq API Error: {e}")
        return "Description unavailable due to Groq API error."


def process_audio(y, sr=44100):
    """Clean and normalize audio using spectral gating and bandpass filtering."""
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
        print(f"Audio processing error: {e}")
        return y


def fix_audio_with_sox(original_path):
    """Attempt to repair audio files using SOX."""
    try:
        fixed_path = original_path.replace(".wav", "_fixed.wav")
        subprocess.run(['sox', original_path, '-c', '1', '-r', '44100', fixed_path], check=True)
        return fixed_path
    except Exception as e:
        print(f"SOX audio repair failed: {e}")
        return None


def create_all_visualizations(y, sr, species=None):
    """Generate a 4-panel audio visualization figure."""
    plt.style.use('dark_background')
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), facecolor='#121212')

    time = np.linspace(0, len(y)/sr, len(y))
    axs[0].plot(time, y, color='cyan', linewidth=0.5)
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


def analyze_bird_sound(audio_path, min_conf=0.2):
    """
    Main function to analyze bird sound from an audio file.
    
    Returns:
        results: List of detections with species info and Groq-generated descriptions
        y_processed: Processed audio waveform
        sr: Sample rate
        feature_summary: ML feature extraction summary
    """
    try:
        # Load and process audio
        y, sr = librosa.load(audio_path, sr=44100)
        y_processed = process_audio(y, sr)
        
        # Extract ML features
        feature_summary = feature_extractor.get_feature_summary(y_processed)
        
        # Save processed audio for BirdNet
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, y_processed, 44100)
            processed_path = tmp.name

        analyzer = Analyzer()
        rec = Recording(analyzer=analyzer, path=processed_path, date=datetime.now(), min_conf=min_conf, overlap=1.0)
        rec.analyze()

        results = []
        for det in rec.detections:
            if 'common_name' in det and det['confidence'] >= min_conf:
                description = get_bird_description_from_groq(det['common_name'])
                results.append({
                    "Species": det['common_name'],
                    "Scientific Name": det['scientific_name'],
                    "Confidence": det['confidence'],
                    "Description": description
                })
        
        # Cleanup
        os.unlink(processed_path)
        
        return results, y_processed, sr, feature_summary

    except Exception as e:
        print(f"Analysis error: {e}")
        return [], None, 44100, {}
