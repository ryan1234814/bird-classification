"""
Neural Network Module for Bird Audio Classification
=====================================================
Implements a custom Convolutional Neural Network (CNN) using PyTorch for
classifying bird species from audio spectrograms.

Architecture:
    - 3 Convolutional Blocks (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
    - Adaptive Average Pooling (handles variable-length inputs)
    - Fully Connected Classifier Head (FC → ReLU → Dropout → FC)

The model takes a 3-channel spectrogram tensor as input:
    Channel 0: Mel spectrogram (frequency content)
    Channel 1: MFCCs (timbral features)
    Channel 2: Chroma + Spectral Contrast (pitch & harmonic features)

This module also provides an AudioClassificationPipeline class that integrates
feature extraction, training, inference, and model persistence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple

from model.feature_extractor import AudioFeatureExtractor


class BirdAudioCNN(nn.Module):
    """
    Convolutional Neural Network for bird species classification from
    audio spectrograms.
    
    The architecture is designed to progressively extract hierarchical features
    from multi-channel spectrogram inputs:
    
    Block 1: Low-level features (edges, simple patterns in spectrograms)
    Block 2: Mid-level features (harmonic structures, temporal patterns)
    Block 3: High-level features (species-specific call signatures)
    
    Parameters
    ----------
    num_classes : int
        Number of bird species to classify (default: 100).
    in_channels : int
        Number of input channels (default: 3 — mel, mfcc, chroma+contrast).
    dropout_rate : float
        Dropout probability for regularization (default: 0.3).
    """

    def __init__(self, num_classes: int = 100, in_channels: int = 3, dropout_rate: float = 0.3):
        super(BirdAudioCNN, self).__init__()

        self.num_classes = num_classes

        # === Convolutional Block 1 ===
        # Input: (batch, 3, 128, T)
        # Output: (batch, 32, 64, T/2)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        # === Convolutional Block 2 ===
        # Output: (batch, 64, 32, T/4)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        # === Convolutional Block 3 ===
        # Output: (batch, 128, 16, T/8)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(dropout_rate),
        )

        # === Global Pooling ===
        # Adaptive pooling ensures fixed output regardless of input time dimension
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # === Fully Connected Classification Head ===
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3, 128, T)
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, num_classes)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 128-dimensional feature embedding from the penultimate
        layer. Useful for transfer learning, similarity search, or
        visualization with t-SNE/UMAP.
        
        Returns: Tensor of shape (batch, 128)
        """
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # Pass through the classifier up to the penultimate layer
        x = self.classifier[1](x)   # Linear(2048, 256)
        x = self.classifier[2](x)   # ReLU
        x = self.classifier[3](x)   # Dropout
        x = self.classifier[4](x)   # Linear(256, 128)
        return x

    def get_model_summary(self) -> Dict:
        """Return a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": "BirdAudioCNN",
            "num_classes": self.num_classes,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "conv_blocks": 3,
            "embedding_dim": 128,
            "input_shape": "(batch, 3, 128, T)",
        }


class BirdAudioDataset(Dataset):
    """
    PyTorch Dataset for bird audio spectrograms.
    
    Wraps pre-extracted tensors and labels for use with DataLoader.
    """

    def __init__(self, spectrograms: List[torch.Tensor], labels: List[int]):
        self.spectrograms = spectrograms
        self.labels = labels

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]


class AudioClassificationPipeline:
    """
    End-to-end pipeline for bird audio classification.
    
    Integrates:
    - AudioFeatureExtractor for feature/spectrogram extraction
    - BirdAudioCNN for deep learning classification
    - Training loop with loss tracking
    - Inference with top-k predictions
    - Model persistence (save/load)
    
    Parameters
    ----------
    num_classes : int
        Number of species classes.
    model_dir : str
        Directory for saving/loading model weights and metadata.
    device : str
        Compute device ('cpu', 'cuda', or 'mps').
    """

    def __init__(
        self,
        num_classes: int = 100,
        model_dir: str = None,
        device: str = None,
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.num_classes = num_classes
        self.model = BirdAudioCNN(num_classes=num_classes).to(self.device)
        self.feature_extractor = AudioFeatureExtractor()
        self.species_labels: List[str] = []
        self.is_trained = False

        # Model directory for persistence
        if model_dir is None:
            self.model_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "saved_models"
            )
        else:
            self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(
        self,
        audio_data: List[np.ndarray],
        labels: List[int],
        species_names: List[str],
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.001,
        validation_split: float = 0.2,
    ) -> Dict:
        """
        Train the CNN model on audio data.
        
        Parameters
        ----------
        audio_data : list of np.ndarray
            Raw audio waveforms.
        labels : list of int
            Integer class labels for each audio sample.
        species_names : list of str
            Human-readable species names corresponding to class indices.
        epochs : int
            Number of training epochs.
        batch_size : int
            Training batch size.
        learning_rate : float
            Optimizer learning rate.
        validation_split : float
            Fraction of data to use for validation.
        
        Returns
        -------
        dict
            Training history with loss and accuracy curves.
        """
        self.species_labels = species_names
        self.model.num_classes = len(species_names)

        # Extract features
        print("[Pipeline] Extracting features from audio data...")
        spectrograms = []
        for audio in audio_data:
            spec = self.feature_extractor.get_cnn_input(audio)
            spectrograms.append(spec)

        # Train/validation split
        n_val = int(len(spectrograms) * validation_split)
        indices = np.random.permutation(len(spectrograms))
        train_idx, val_idx = indices[n_val:], indices[:n_val]

        train_specs = [spectrograms[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_specs = [spectrograms[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_dataset = BirdAudioDataset(train_specs, train_labels)
        val_dataset = BirdAudioDataset(val_specs, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        print(f"[Pipeline] Training on {len(train_specs)} samples, validating on {len(val_specs)}")
        print(f"[Pipeline] Device: {self.device}")

        self.model.train()
        for epoch in range(epochs):
            # --- Training ---
            train_loss, train_correct, train_total = 0.0, 0, 0
            for batch_specs, batch_labels in train_loader:
                batch_specs = batch_specs.to(self.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_specs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_specs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
                train_total += batch_specs.size(0)

            # --- Validation ---
            val_loss, val_correct, val_total = 0.0, 0, 0
            self.model.eval()
            with torch.no_grad():
                for batch_specs, batch_labels in val_loader:
                    batch_specs = batch_specs.to(self.device)
                    batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                    outputs = self.model(batch_specs)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item() * batch_specs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
                    val_total += batch_specs.size(0)
            self.model.train()

            epoch_train_loss = train_loss / max(train_total, 1)
            epoch_val_loss = val_loss / max(val_total, 1)
            epoch_train_acc = train_correct / max(train_total, 1)
            epoch_val_acc = val_correct / max(val_total, 1)

            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)

            scheduler.step(epoch_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {epoch_train_loss:.4f} | "
                    f"Val Loss: {epoch_val_loss:.4f} | "
                    f"Train Acc: {epoch_train_acc:.3f} | "
                    f"Val Acc: {epoch_val_acc:.3f}"
                )

        self.is_trained = True
        print("[Pipeline] Training complete.")
        return history

    def predict(self, audio: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Predict bird species from an audio waveform.
        
        Parameters
        ----------
        audio : np.ndarray
            Raw audio waveform.
        top_k : int
            Number of top predictions to return.
        
        Returns
        -------
        list of dict
            Top-k predictions with species name and confidence score.
        """
        self.model.eval()

        spec = self.feature_extractor.get_cnn_input(audio).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(spec)
            probabilities = torch.softmax(logits, dim=1).squeeze(0)

        top_probs, top_indices = torch.topk(probabilities, min(top_k, self.num_classes))

        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            species = self.species_labels[idx] if idx < len(self.species_labels) else f"class_{idx}"
            predictions.append({
                "species": species,
                "confidence": float(prob),
                "class_index": int(idx),
            })

        return predictions

    def get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract the 128-dim embedding vector for an audio sample.
        Useful for similarity search or clustering.
        """
        self.model.eval()
        spec = self.feature_extractor.get_cnn_input(audio).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.extract_embeddings(spec)
        return embedding.squeeze(0).cpu().numpy()

    def save_model(self, name: str = "bird_cnn"):
        """Save model weights and metadata."""
        model_path = os.path.join(self.model_dir, f"{name}.pth")
        meta_path = os.path.join(self.model_dir, f"{name}_meta.json")

        torch.save(self.model.state_dict(), model_path)

        meta = {
            "num_classes": self.num_classes,
            "species_labels": self.species_labels,
            "is_trained": self.is_trained,
            "model_summary": self.model.get_model_summary(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[Pipeline] Model saved to {model_path}")

    def load_model(self, name: str = "bird_cnn") -> bool:
        """Load model weights and metadata. Returns True if successful."""
        model_path = os.path.join(self.model_dir, f"{name}.pth")
        meta_path = os.path.join(self.model_dir, f"{name}_meta.json")

        if not os.path.exists(model_path):
            print(f"[Pipeline] No saved model found at {model_path}")
            return False

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)

            self.num_classes = meta["num_classes"]
            self.species_labels = meta["species_labels"]
            self.is_trained = meta["is_trained"]

            self.model = BirdAudioCNN(num_classes=self.num_classes).to(self.device)
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )
            self.model.eval()

            print(f"[Pipeline] Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"[Pipeline] Error loading model: {e}")
            return False

    def get_pipeline_info(self) -> Dict:
        """Return information about the pipeline configuration."""
        return {
            "device": str(self.device),
            "model_summary": self.model.get_model_summary(),
            "feature_extractor": {
                "sample_rate": self.feature_extractor.sr,
                "n_mfcc": self.feature_extractor.n_mfcc,
                "n_mels": self.feature_extractor.n_mels,
                "n_fft": self.feature_extractor.n_fft,
                "hop_length": self.feature_extractor.hop_length,
                "max_duration_sec": self.feature_extractor.max_duration,
            },
            "is_trained": self.is_trained,
            "num_species": len(self.species_labels),
            "model_dir": self.model_dir,
        }
