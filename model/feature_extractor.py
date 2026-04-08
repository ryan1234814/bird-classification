"""
Audio Feature Extraction Module
================================
Implements a comprehensive deep audio feature extraction pipeline for bird
sound classification. Extracts multiple feature types commonly used in
audio ML/DL research:

- MFCC (Mel-Frequency Cepstral Coefficients)
- Mel Spectrograms
- Spectral Contrast
- Chroma Features
- Zero-Crossing Rate
- Spectral Centroid & Bandwidth
- Spectral Rolloff
- RMS Energy
- Delta and Delta-Delta features

All features are extracted using librosa and returned as both raw numpy arrays
and PyTorch tensors for CNN input.
"""

import numpy as np
import librosa
import torch
from typing import Dict, Tuple, Optional


class AudioFeatureExtractor:
    """
    Comprehensive audio feature extractor for bird sound classification.
    
    Extracts a rich set of audio features suitable for both traditional ML
    (flattened feature vectors) and deep learning (spectrograms/tensors).
    
    Parameters
    ----------
    sr : int
        Target sample rate for audio processing (default: 44100 Hz).
    n_mfcc : int
        Number of MFCC coefficients to extract (default: 40).
    n_mels : int
        Number of mel filter banks for mel spectrogram (default: 128).
    n_fft : int
        FFT window size (default: 2048).
    hop_length : int
        Hop length between frames (default: 512).
    max_duration : float
        Maximum audio duration in seconds. Longer clips are truncated,
        shorter clips are zero-padded (default: 10.0s).
    """

    def __init__(
        self,
        sr: int = 44100,
        n_mfcc: int = 40,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        max_duration: float = 10.0,
    ):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_duration = max_duration
        self.max_samples = int(sr * max_duration)

    def _normalize_length(self, y: np.ndarray) -> np.ndarray:
        """Pad or truncate audio to a fixed length for consistent tensor shapes."""
        if len(y) > self.max_samples:
            return y[:self.max_samples]
        elif len(y) < self.max_samples:
            return np.pad(y, (0, self.max_samples - len(y)), mode='constant')
        return y

    def extract_mfcc(self, y: np.ndarray) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCCs).
        
        MFCCs capture the timbral characteristics of audio and are the most
        widely used features in audio classification tasks.
        
        Returns: (n_mfcc, time_frames) array
        """
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=self.n_mfcc,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        return mfcc

    def extract_mfcc_deltas(self, mfcc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute delta and delta-delta (acceleration) of MFCCs.
        
        Delta features capture the temporal dynamics of the audio signal,
        which is critical for distinguishing bird call patterns.
        
        Returns: (delta, delta2) tuple of arrays
        """
        delta = librosa.feature.delta(mfcc, order=1)
        delta2 = librosa.feature.delta(mfcc, order=2)
        return delta, delta2

    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """
        Compute the mel-scaled spectrogram in decibels.
        
        Mel spectrograms are the primary input for CNN-based audio classifiers.
        The mel scale approximates human auditory perception.
        
        Returns: (n_mels, time_frames) array in dB scale
        """
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_mels=self.n_mels,
            n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db

    def extract_spectral_contrast(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral contrast features (7 frequency bands).
        
        Spectral contrast measures the difference between peaks and valleys
        in the spectrum — useful for distinguishing harmonic bird calls
        from noisy backgrounds.
        
        Returns: (7, time_frames) array
        """
        contrast = librosa.feature.spectral_contrast(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return contrast

    def extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Extract chroma (pitch class) features.
        
        Chroma features represent the energy distribution across the 12 pitch
        classes, capturing melodic patterns in bird songs.
        
        Returns: (12, time_frames) array
        """
        chroma = librosa.feature.chroma_stft(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        return chroma

    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract additional spectral descriptors.
        
        Returns a dictionary with:
        - spectral_centroid: Center of mass of the spectrum
        - spectral_bandwidth: Spread of the spectrum
        - spectral_rolloff: Frequency below which 85% of energy is concentrated
        - zero_crossing_rate: Rate of sign changes in the signal
        - rms_energy: Root-mean-square energy of the signal
        """
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )
        zcr = librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop_length
        )
        rms = librosa.feature.rms(
            y=y, frame_length=self.n_fft, hop_length=self.hop_length
        )

        return {
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "zero_crossing_rate": zcr,
            "rms_energy": rms,
        }

    def extract_all_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract the complete feature set from an audio signal.
        
        Returns a dictionary containing all feature types with their
        raw multi-dimensional arrays.
        """
        y = self._normalize_length(y)

        mfcc = self.extract_mfcc(y)
        delta, delta2 = self.extract_mfcc_deltas(mfcc)
        mel_spec = self.extract_mel_spectrogram(y)
        contrast = self.extract_spectral_contrast(y)
        chroma = self.extract_chroma(y)
        spectral = self.extract_spectral_features(y)

        return {
            "mfcc": mfcc,
            "mfcc_delta": delta,
            "mfcc_delta2": delta2,
            "mel_spectrogram": mel_spec,
            "spectral_contrast": contrast,
            "chroma": chroma,
            **spectral,
        }

    def get_feature_vector(self, y: np.ndarray) -> np.ndarray:
        """
        Compute a fixed-size feature vector by aggregating temporal statistics.
        
        For each feature matrix (n_features, time_frames), computes:
        - Mean across time
        - Standard deviation across time
        
        This produces a fixed-length vector suitable for traditional ML models
        (SVM, Random Forest, etc.) regardless of audio duration.
        
        Returns: 1D numpy array of aggregated features
        """
        features = self.extract_all_features(y)
        
        aggregated = []
        for name, feat in features.items():
            feat = np.atleast_2d(feat)
            aggregated.append(np.mean(feat, axis=1).flatten())
            aggregated.append(np.std(feat, axis=1).flatten())

        return np.concatenate(aggregated)

    def get_cnn_input(self, y: np.ndarray) -> torch.Tensor:
        """
        Prepare a mel spectrogram tensor for CNN input.
        
        Stacks mel spectrogram, MFCC, and chroma into a 3-channel image-like
        tensor, analogous to RGB channels in image classification.
        
        Channel 0: Mel spectrogram (128 bins) — frequency content
        Channel 1: MFCCs (40 coefficients, padded to 128) — timbral features
        Channel 2: Chroma + contrast (12 + 7 = 19, padded to 128) — pitch & harmonic features
        
        Returns: Tensor of shape (3, 128, time_frames)
        """
        y = self._normalize_length(y)

        mel_spec = self.extract_mel_spectrogram(y)       # (128, T)
        mfcc = self.extract_mfcc(y)                       # (40, T)
        chroma = self.extract_chroma(y)                   # (12, T)
        contrast = self.extract_spectral_contrast(y)      # (7, T)

        T = mel_spec.shape[1]

        # Pad MFCC from 40 -> 128 rows
        mfcc_padded = np.zeros((self.n_mels, T))
        mfcc_padded[:self.n_mfcc, :] = mfcc

        # Combine chroma (12) + contrast (7) = 19, pad to 128
        chroma_contrast = np.vstack([chroma, contrast])   # (19, T)
        cc_padded = np.zeros((self.n_mels, T))
        cc_padded[:chroma_contrast.shape[0], :] = chroma_contrast

        # Normalize each channel to [0, 1]
        def normalize_channel(ch):
            ch_min, ch_max = ch.min(), ch.max()
            if ch_max - ch_min > 1e-8:
                return (ch - ch_min) / (ch_max - ch_min)
            return np.zeros_like(ch)

        stacked = np.stack([
            normalize_channel(mel_spec),
            normalize_channel(mfcc_padded),
            normalize_channel(cc_padded),
        ])  # (3, 128, T)

        return torch.FloatTensor(stacked)

    def get_feature_summary(self, y: np.ndarray) -> Dict:
        """
        Generate a human-readable summary of extracted features.
        
        Useful for the API to return feature analysis results to the frontend.
        """
        features = self.extract_all_features(y)

        summary = {
            "feature_shapes": {k: list(v.shape) for k, v in features.items()},
            "mfcc_stats": {
                "mean": float(np.mean(features["mfcc"])),
                "std": float(np.std(features["mfcc"])),
                "min": float(np.min(features["mfcc"])),
                "max": float(np.max(features["mfcc"])),
            },
            "mel_spectrogram_stats": {
                "mean": float(np.mean(features["mel_spectrogram"])),
                "std": float(np.std(features["mel_spectrogram"])),
                "dynamic_range_db": float(
                    np.max(features["mel_spectrogram"]) - np.min(features["mel_spectrogram"])
                ),
            },
            "spectral_centroid_mean_hz": float(np.mean(features["spectral_centroid"])),
            "spectral_bandwidth_mean_hz": float(np.mean(features["spectral_bandwidth"])),
            "zero_crossing_rate_mean": float(np.mean(features["zero_crossing_rate"])),
            "rms_energy_mean": float(np.mean(features["rms_energy"])),
            "feature_vector_length": len(self.get_feature_vector(y)),
            "total_parameters_extracted": int(sum(
                np.prod(v.shape) for v in features.values()
            )),
        }
        return summary
