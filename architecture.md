# Project Architecture: Bird Identification System v2.0

This document outlines the technical architecture of the Bird Identification System, a multi-stage machine learning pipeline and modern web application for avian acoustic analysis.

## 1. System Overview

The system is designed as a decoupled architecture consisting of a **React-based Frontend** and a **FastAPI-powered Backend**. 

### High-Level Flow:
1. **Input**: Audio is captured via the browser microphone or uploaded as a file.
2. **Preprocessing**: The backend uses `ffmpeg` and `librosa` to normalize the audio (44.1kHz, mono, noise gating).
3. **Analysis Stage 1 (BirdNet)**: A pre-trained BirdNet-Analyzer identifies likely species candidates.
4. **Analysis Stage 2 (Custom CNN)**: A custom PyTorch Convolutional Neural Network extracts deep audio features (Mel Spectrogram, MFCC, Chroma) and provides complementary species insights.
5. **Knowledge Enrichment (Groq LLM)**: The system queries the `llama-3.3-70b-versatile` model via Groq API to generate structured habitat, diet, and geographic data.
6. **Output**: Results are visualized in a responsive "Nature-Tech" UI with interactive OpenStreetMap integration.

---

## 2. Backend Architecture (Python / FastAPI)

Located in `/backend` and `/model`.

### Key Modules:
- **`backend/main.py`**: The API gateway. Handles file uploads, asynchronous processing, and defines the REST endpoints (`/analyze`, `/features`).
- **`model/classifier.py`**: The orchestration engine. Coordinates the BirdNet analyzer, custom model inference, and LLM calls.
- **`model/feature_extractor.py`**: A dedicated signal processing module that converts raw waveforms into multi-channel tensors (Mel + MFCC + Contrast).
- **`model/neural_network.py`**: PyTorch implementation of the `BirdAudioCNN` architecture.

### AI Stack:
- **BirdNet-Analyzer**: Used for broad species detection across thousands of classes.
- **Groq API**: Used for generating structured JSON descriptions.
- **PyTorch**: Powers the custom CNN classification pipeline.

---

## 3. Frontend Architecture (React / Vite)

Located in `/frontend`.

### UI Components:
- **Nature-Tech Design System**: Built with modern CSS (glassmorphism, vibrant forest gradients) and **Framer Motion** for animations.
- **`SpotlightCard.jsx`**: Dynamically highlights top-confidence matches with celebratory effects.
- **`AudioVisualizer.jsx`**: Real-time Web Audio API visualization for live recording feedback.
- **`App.jsx`**: Local state management using React hooks (`useState`, `useRef`, `useEffect`) and `react-leaflet` for geographic visualization.

### Third-Party Libraries:
- **React-Leaflet**: OpenStreetMap integration for species distribution mapping.
- **Lucide-React**: Harmonious icon system for nature-tech visual language.
- **Axios**: Promised-based HTTP client for API communication.
- **Canvas-Confetti**: UI micro-interactions for high-confidence detections.

---

## 4. Geographic Intelligence

The system utilizes an enhanced OpenStreetMap integration that goes beyond simple ranges:
- **Primary Habitiat**: Markings for the geographic center of the bird's distribution.
- **Conservation Hotspots**: AI-identified "Major Places" (National Parks/Reserves) plotted on the map with custom popups describing their significance.

---

## 5. Directory Structure
```text
.
├── backend/                # FastAPI Application
├── frontend/               # React Native/Web Application
│   ├── src/
│   │   ├── components/     # UI Components (Spotlight, Visualizer)
│   │   └── App.jsx         # Main Logic
├── model/                  # AI Model Logic
│   ├── classifier.py       # Core Orchestrator
│   ├── neural_network.py   # PyTorch CNN
│   └── feature_extractor.py# Signal Processing
├── architecture.md         # This Document
└── docker-compose.yml      # Docker Orchestration

---

## 6. Containerization (Docker)

The application is fully containerized for consistent deployment:
- **Backend Image**: Based on `python:3.11-slim`, with `ffmpeg` and `libsndfile` pre-installed.
- **Frontend Image**: Based on `node:20-slim`, serving the Vite development environment.
- **Orchestration**: Managed via `docker-compose.yml`, which handles volume mapping for hot-reloading and environment variable injection.
```
