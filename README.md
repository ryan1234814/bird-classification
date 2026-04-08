# 🦜 Bird Sound Identifier

A full-stack ML/DL web application that identifies bird species from audio recordings. It combines **BirdNet** (pre-trained deep learning) with a **custom PyTorch CNN** for audio classification, and uses **Groq LLM** (Llama 3.3 70B) for generating detailed species descriptions.

## 🚀 Containerization

If you have Docker installed, you can launch the entire stack with a single command:

```bash
docker-compose up --build
```

Access the UI at `http://localhost:5173` and the API at `http://localhost:8000`.

## 🧠 ML/DL Architecture

```
Audio Input
    │
    ├─── Audio Preprocessing (librosa + scipy)
    │        ├── Spectral Noise Gating
    │        └── Butterworth Bandpass Filter (300–8000 Hz)
    │
    ├─── BirdNet Analysis (Pre-trained Deep Learning)
    │        └── Species detection + confidence scores
    │
    ├─── Custom ML/DL Pipeline
    │        ├── Feature Extraction (model/feature_extractor.py)
    │        │     ├── MFCCs (40 coefficients + deltas)
    │        │     ├── Mel Spectrograms (128 mel bins)
    │        │     ├── Spectral Contrast (7 bands)
    │        │     ├── Chroma Features (12 pitch classes)
    │        │     ├── Spectral Centroid & Bandwidth
    │        │     ├── Zero-Crossing Rate
    │        │     └── RMS Energy
    │        └── CNN Classifier (model/neural_network.py)
    │              ├── 3 Conv Blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
    │              ├── Adaptive Average Pooling
    │              └── FC Classification Head (256 → 128 → num_classes)
    │
    ├─── Groq LLM (llama-3.3-70b-versatile)
    │        └── AI-generated species descriptions
    │
    └─── Visualization (matplotlib)
             └── Waveform, Spectrogram, Mel Spectrogram, Chromagram
```

## 🏗️ Project Structure

```
bird_recognition/
├── .env                              # Groq API key (environment config)
├── README.md
├── model/
│   ├── __init__.py                   # Package exports
│   ├── b.py                          # Standalone analysis script
│   ├── classifier.py                 # Main BirdClassifier class
│   ├── feature_extractor.py          # Deep audio feature extraction (ML)
│   ├── neural_network.py             # PyTorch CNN model (DL)
│   └── saved_models/                 # Trained model weights (auto-created)
├── backend/
│   ├── main.py                       # FastAPI server
│   └── requirements.txt              # Python dependencies
└── frontend/
    ├── src/
    │   ├── App.jsx                   # React UI
    │   └── index.css                 # Premium dark-mode styles
    └── package.json
```

## 🚀 Execution Guide

### 1. Prerequisites
- Python 3.9+
- Node.js & npm
- FFmpeg (Required for audio processing: `brew install ffmpeg`)
- SOX (Recommended: `brew install sox`)

### 2. Environment Configuration
Create a `.env` file in the project root (already included):
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Backend Setup
```bash
# Activate the Python 3.11 virtual environment
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the server
cd backend
python main.py
```
*The backend will run on `http://localhost:8000`.*

**API Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/analyze` | POST | Full bird classification pipeline |
| `/features` | POST | ML feature extraction only |
| `/model-info` | GET | CNN model & pipeline information |

### 4. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*The frontend will run on `http://localhost:5173`.*

---

## 🛠️ Key Features

- **Live Recording**: Record bird sounds directly from your browser.
- **File Upload**: Support for `.wav`, `.mp3`, `.ogg`, and `.m4a` files.
- **Dual Classification**: BirdNet (pre-trained DL) + Custom CNN (PyTorch).
- **Deep Feature Extraction**: MFCCs, Mel Spectrograms, Spectral Contrast, Chroma, and more.
- **AI Insights**: Groq LLM generates rich species descriptions and habitat info.
- **Visual Analysis**: Waveforms, Spectrograms, Mel Spectrograms, and Chromagrams.
- **Premium Design**: Modern, responsive dark-mode interface with glassmorphism effects.

---

## 🔬 ML/DL Components

### Feature Extractor (`model/feature_extractor.py`)
Extracts a comprehensive set of audio features used in state-of-the-art audio ML research:
- **MFCC**: 40 mel-frequency cepstral coefficients with delta and delta-delta
- **Mel Spectrogram**: 128-bin perceptual frequency representation
- **Spectral Contrast**: Peak vs. valley energy ratios across 7 bands
- **Chroma**: Energy distribution across 12 pitch classes
- Outputs both raw arrays (for visualization) and PyTorch tensors (for CNN input)

### Neural Network (`model/neural_network.py`)
Custom **BirdAudioCNN** — a 3-block CNN built with PyTorch:
- ~200K trainable parameters
- Takes 3-channel spectrogram input (mel + MFCC + chroma/contrast)
- Includes full training pipeline with validation, learning rate scheduling
- Embedding extraction for transfer learning and similarity search
- Model save/load for persistence

### Classifier (`model/classifier.py`)
Unified **BirdClassifier** that integrates all components:
- Audio preprocessing (noise gating, bandpass filtering)
- BirdNet analysis
- Custom CNN inference
- Groq LLM description generation
- Visualization generation

---

## 📝 Important Notes

- **API Configuration**: The Groq API key is loaded from the `.env` file. Never commit this file to version control.
- **BirdNet Model**: On the first run, the backend may download BirdNet model files (~500MB).
- **CNN Training**: The custom CNN architecture is ready for training. When training data is available, use the `AudioClassificationPipeline.train()` method.
- **GPU Support**: PyTorch automatically uses CUDA or MPS (Apple Silicon) if available.
