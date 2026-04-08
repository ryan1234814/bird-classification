from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sys
import os
import shutil
import tempfile
import subprocess

# Add the parent directory to sys.path to import the model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.classifier import BirdClassifier

app = FastAPI(
    title="Bird Sound Identifier API",
    description=(
        "ML/DL-powered bird species identification from audio. "
        "Uses BirdNet (pre-trained DL), a custom PyTorch CNN, "
        "and Groq LLM for species descriptions."
    ),
    version="2.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the actual origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the classifier
classifier = BirdClassifier()


def convert_to_wav(input_path: str) -> str:
    """
    Convert any audio file to a proper 44100Hz mono WAV using ffmpeg.
    
    The browser's MediaRecorder records in webm/opus format, which librosa
    cannot read natively. This function converts ANY audio format to a
    standard WAV that librosa can process without errors.
    
    Returns the path to the converted WAV file.
    """
    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",         # Overwrite output
                "-i", input_path,       # Input file
                "-ar", "44100",         # Sample rate: 44100 Hz
                "-ac", "1",             # Mono channel
                "-sample_fmt", "s16",   # 16-bit PCM
                "-f", "wav",            # Output format: WAV
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"ffmpeg conversion warning: {result.stderr[:500]}")
            # If ffmpeg fails, return original path as fallback
            return input_path
        return output_path
    except FileNotFoundError:
        print("WARNING: ffmpeg not found. Audio conversion skipped.")
        return input_path
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return input_path


@app.get("/")
async def root():
    return {
        "message": "Bird Sound Identifier API is running",
        "version": "2.0.0",
        "ml_pipeline": "BirdNet + Custom CNN (PyTorch) + Groq LLM",
    }


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file for bird species identification.
    
    Accepts any audio format (wav, mp3, webm, ogg, m4a, etc.).
    Automatically converts to WAV before processing.
    """
    # Accept any file — the browser may send audio/webm, audio/ogg, etc.
    tmp_path = None
    wav_path = None

    try:
        # Determine the file extension from content type or filename
        ext = ".wav"
        if file.filename:
            ext = os.path.splitext(file.filename)[1] or ext
        elif file.content_type:
            type_map = {
                "audio/webm": ".webm",
                "audio/ogg": ".ogg",
                "audio/mp3": ".mp3",
                "audio/mpeg": ".mp3",
                "audio/mp4": ".m4a",
                "audio/x-m4a": ".m4a",
                "audio/wav": ".wav",
                "audio/x-wav": ".wav",
            }
            ext = type_map.get(file.content_type, ".webm")

        # Save uploaded file with its original extension
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Convert to proper WAV (handles webm, ogg, mp3, etc.)
        wav_path = convert_to_wav(tmp_path)

        # Run classification
        results = classifier.classify(wav_path)

        return results

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error during analysis:\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Analysis Engine Error: {str(e)}")

    finally:
        # Cleanup all temp files
        for path in [tmp_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.post("/features")
async def extract_features(file: UploadFile = File(...)):
    """
    Extract ML/DL features from an uploaded audio file.
    
    Accepts any audio format. Returns a comprehensive feature analysis
    including MFCCs, mel spectrogram statistics, spectral features, and
    CNN pipeline information.
    """
    tmp_path = None
    wav_path = None

    try:
        ext = ".wav"
        if file.filename:
            ext = os.path.splitext(file.filename)[1] or ext

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Convert to proper WAV
        wav_path = convert_to_wav(tmp_path)

        # Extract features
        feature_data = classifier.extract_features(wav_path)

        return feature_data

    except Exception as e:
        print(f"Error during feature extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for path in [tmp_path, wav_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


@app.get("/model-info")
async def model_info():
    """
    Get information about the ML/DL pipeline configuration.
    
    Returns model architecture details, feature extractor config,
    and training status.
    """
    return classifier.cnn_pipeline.get_pipeline_info()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
