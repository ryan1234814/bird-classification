"""
Bird Recognition Model Package
===============================
Provides ML/DL-powered bird audio classification with:
- BirdNet pre-trained deep learning analysis
- Custom CNN-based audio classifier (PyTorch)
- Deep audio feature extraction (MFCCs, Mel Spectrograms, Chroma, etc.)
- Groq LLM-powered species description generation
"""

from model.classifier import BirdClassifier
from model.feature_extractor import AudioFeatureExtractor
from model.neural_network import BirdAudioCNN, AudioClassificationPipeline

__all__ = [
    "BirdClassifier",
    "AudioFeatureExtractor",
    "BirdAudioCNN",
    "AudioClassificationPipeline",
]
