"""
Bird Classifier Module
=======================
Main classification engine that integrates:
1. BirdNet — pre-trained DL model for bird species identification
2. Custom CNN Pipeline — PyTorch-based audio classification
3. Audio Feature Extraction — MFCCs, Mel Spectrograms, etc.
4. Groq LLM — AI-powered species description generation

The BirdClassifier class provides a unified API for the backend to call.
"""

import numpy as np
import librosa
import scipy.signal
from groq import Groq
from birdnetlib.analyzer import Analyzer
from birdnetlib import Recording
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import librosa.display
import io
import base64
import os
import warnings
from dotenv import load_dotenv

from model.feature_extractor import AudioFeatureExtractor
from model.neural_network import AudioClassificationPipeline

warnings.filterwarnings('ignore')

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))


class BirdClassifier:
    """
    Unified bird sound classifier combining BirdNet, a custom CNN pipeline,
    and Groq LLM for description generation.
    
    Parameters
    ----------
    groq_api_key : str, optional
        Groq API key. If not provided, reads from GROQ_API_KEY env var.
    """

    def __init__(self, groq_api_key: str = None):
        # --- Groq LLM Client ---
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Groq API key not found. Set GROQ_API_KEY environment variable "
                "or pass groq_api_key parameter."
            )
        self.groq_client = Groq(api_key=api_key)
        self.groq_model = "llama-3.3-70b-versatile"

        # --- BirdNet Analyzer (pre-trained DL model) ---
        print("[BirdClassifier] Loading BirdNet analyzer...")
        self.analyzer = Analyzer()

        # --- Custom ML/DL Pipeline ---
        print("[BirdClassifier] Initializing custom CNN pipeline...")
        self.feature_extractor = AudioFeatureExtractor()
        self.cnn_pipeline = AudioClassificationPipeline()
        # Attempt to load a previously trained model
        self.cnn_pipeline.load_model()

        print("[BirdClassifier] Initialization complete.")

    def process_audio(self, y: np.ndarray, sr: int = 44100) -> np.ndarray:
        """
        Clean and normalize audio using spectral gating and bandpass filtering.
        
        Steps:
        1. STFT decomposition
        2. Spectral noise gating (20th percentile threshold)
        3. ISTFT reconstruction
        4. 6th-order Butterworth bandpass filter (300–8000 Hz)
        5. Peak normalization
        """
        try:
            D = librosa.stft(y)
            magnitude = np.abs(D)
            phase = np.angle(D)
            noise_threshold = np.percentile(magnitude, 20, axis=1, keepdims=True)
            mask = magnitude > (noise_threshold * 1.5)
            y_clean = librosa.istft(magnitude * mask * np.exp(1j * phase))
            sos = scipy.signal.butter(6, [300 / (sr / 2), 8000 / (sr / 2)], btype='band', output='sos')
            return librosa.util.normalize(scipy.signal.sosfiltfilt(sos, y_clean))
        except Exception as e:
            print(f"Audio processing error: {e}")
            return y

    def get_bird_description(self, bird_name: str) -> dict:
        """
        Generate a detailed, structured bird species description using Groq LLM.
        
        Uses the llama-3.3-70b-versatile model via Groq API. Explictly requests
        a JSON object for frontend rendering.
        """
        import json
        
        default_error_resp = {
            "overview": "Description unavailable due to API error.",
            "physical_characteristics": "",
            "habitat": "",
            "diet": "",
            "distribution_regions": "",
            "map_coordinates": {"lat": 0.0, "lng": 0.0, "zoom": 2},
            "major_places": []
        }
        
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert ornithologist. Output responses in JSON format exclusively. "
                            "The JSON must have this exact schema: "
                            "{\"overview\": \"str\", \"physical_characteristics\": \"str\", "
                            "\"habitat\": \"str\", \"diet\": \"str\", \"distribution_regions\": \"str\", "
                            "\"map_coordinates\": {\"lat\": float, \"lng\": float, \"zoom\": int}, "
                            "\"major_places\": [{\"name\": \"str\", \"lat\": float, \"lng\": float, \"description\": \"str\"}] }. "
                            "Provide accurate, informative details. For map_coordinates, provide coordinates "
                            "for the geographic center of the bird's primary habitat. "
                            "In 'major_places', identify 3-5 specific major national parks, nature reserves, or geographic landmarks "
                            "where this bird is frequently observed, including their exact coordinates and a brief note why."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Give a detailed description of the bird: '{bird_name}'"
                    },
                ],
                model=self.groq_model,
                temperature=0.3,
                max_completion_tokens=1000,
                response_format={"type": "json_object"}
            )
            content = chat_completion.choices[0].message.content.strip()
            return json.loads(content)
        except Exception as e:
            print(f"Groq API Error: {e}")
            return default_error_resp

    def create_visualizations(self, y: np.ndarray, sr: int, species: str = None) -> str:
        """
        Generate audio analysis visualizations as a base64-encoded PNG.
        Uses Object-Oriented Matplotlib for better thread safety and reliability.
        """
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            # Create figure object directly to avoid global plt state issues
            fig = Figure(figsize=(12, 14), facecolor='#121212')
            canvas = FigureCanvasAgg(fig)
            axs = fig.subplots(4, 1)

            # Waveform
            time = np.linspace(0, len(y) / sr, len(y))
            axs[0].plot(time, y, color='cyan', linewidth=0.5)
            axs[0].set_title('Waveform', color='white', fontsize=13)
            axs[0].set_xlabel('Time (s)', color='gray')
            axs[0].set_ylabel('Amplitude', color='gray')
            axs[0].tick_params(colors='gray')

            # Log Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img1 = librosa.display.specshow(D, y_axis='log', x_axis='time', sr=sr, ax=axs[1], cmap='magma')
            axs[1].set_title('Spectrogram (Log Scale)', color='white', fontsize=13)
            axs[1].tick_params(colors='gray')

            # Mel Spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            img2 = librosa.display.specshow(
                librosa.power_to_db(mel, ref=np.max),
                y_axis='mel', x_axis='time', sr=sr, ax=axs[2], cmap='plasma'
            )
            axs[2].set_title('Mel Spectrogram', color='white', fontsize=13)
            axs[2].tick_params(colors='gray')

            # Chromagram
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            img3 = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr, ax=axs[3], cmap='cool')
            axs[3].set_title('Chromagram', color='white', fontsize=13)
            axs[3].tick_params(colors='gray')

            if species:
                fig.suptitle(f"Species Spotlight: {species}", color='white', fontsize=16, fontweight='bold', y=0.98)

            fig.tight_layout(pad=3.0)

            # Render to buffer
            buf = io.BytesIO()
            fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), dpi=100)
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            # Important: explicit cleanup
            buf.close()
            
            return img_str
        except Exception as e:
            print(f"[BirdClassifier] Visualization error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def extract_features(self, audio_path: str) -> dict:
        """
        Extract ML features from an audio file.
        
        Returns a summary of all extracted features including shapes,
        statistics, and the computed feature vector length.
        """
        y, sr = librosa.load(audio_path, sr=44100)
        y_processed = self.process_audio(y, sr)

        feature_summary = self.feature_extractor.get_feature_summary(y_processed)
        feature_vector = self.feature_extractor.get_feature_vector(y_processed)

        feature_summary["feature_vector_sample"] = feature_vector[:20].tolist()
        feature_summary["pipeline_info"] = self.cnn_pipeline.get_pipeline_info()

        return feature_summary

    def classify(self, audio_path: str, min_conf: float = 0.2) -> dict:
        """
        Full classification pipeline.
        
        Steps:
        1. Load and preprocess audio (noise gating + bandpass filter)
        2. Run BirdNet analysis (pre-trained DL model)
        3. Extract deep audio features (MFCCs, Mel Spectrogram, etc.)
        4. Run custom CNN inference (if trained)
        5. Generate Groq LLM descriptions for detected species
        6. Create audio visualizations
        
        Returns a dictionary with detections, visualizations, and feature info.
        """
        # Step 1: Load and process audio
        y, sr = librosa.load(audio_path, sr=44100)
        y_processed = self.process_audio(y, sr)

        # Temporarily save processed audio for BirdNet
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            sf.write(tmp.name, y_processed, sr)
            temp_processed_path = tmp.name

        # Step 2: BirdNet Analysis
        rec = Recording(
            analyzer=self.analyzer,
            path=temp_processed_path,
            date=datetime.now(),
            min_conf=min_conf,
            overlap=1.0,
        )
        rec.analyze()

        # Step 3: Extract ML features
        feature_summary = self.feature_extractor.get_feature_summary(y_processed)
        feature_vector = self.feature_extractor.get_feature_vector(y_processed)

        # Step 4: CNN prediction (if model is trained)
        cnn_predictions = []
        if self.cnn_pipeline.is_trained:
            cnn_predictions = self.cnn_pipeline.predict(y_processed, top_k=5)

        # Step 5: Build results with Groq descriptions
        results = []
        for det in rec.detections:
            if 'common_name' in det and det['confidence'] >= min_conf:
                description = self.get_bird_description(det['common_name'])
                results.append({
                    "common_name": det['common_name'],
                    "scientific_name": det['scientific_name'],
                    "confidence": float(det['confidence']),
                    "description": description,
                    "source": "BirdNet",
                })
        
        print(f"[BirdClassifier] Found {len(results)} species.")

        # Cleanup temp file
        if os.path.exists(temp_processed_path):
            os.remove(temp_processed_path)

        # Step 6: Generate visualization
        top_species = results[0]['common_name'] if results else None
        visualization = self.create_visualizations(y_processed, sr, top_species)

        return {
            "detections": results,
            "visualization": visualization,
            "cnn_predictions": cnn_predictions,
            "feature_analysis": {
                "feature_vector_length": len(feature_vector),
                "mfcc_stats": feature_summary.get("mfcc_stats"),
                "mel_spectrogram_stats": feature_summary.get("mel_spectrogram_stats"),
                "spectral_centroid_mean_hz": feature_summary.get("spectral_centroid_mean_hz"),
                "zero_crossing_rate_mean": feature_summary.get("zero_crossing_rate_mean"),
                "rms_energy_mean": feature_summary.get("rms_energy_mean"),
                "total_parameters_extracted": feature_summary.get("total_parameters_extracted"),
            },
            "model_info": self.cnn_pipeline.get_pipeline_info(),
        }
