"""
dataset_loader.py
-----------------
Loads upper + lower STL for a given patient ID and returns a feature vector.

Expected dataset layout:
  dataset/Bits2Bites/
    1/
      upper.stl
      lower.stl
    2/
      upper.stl
      lower.stl
    ...

Augmentation (train=True):
  - Small random jitter on feature values (±2% Gaussian noise)
  - Random sign flip on asymmetry/deviation features to simulate
    mirrored scans
"""

import os
import numpy as np
from preprocessing.stl_features import extract_features

DATASET_ROOT = "dataset/Bits2Bites"

# Indices of features that represent signed lateral deviations
# (can be flipped for mirror augmentation)
_FLIP_INDICES = [3, 4, 19, 20, 39, 40, 43]   # overjet, X centroid, median dev etc.


def load_patient(patient_id: str, train: bool = False) -> np.ndarray:
    """
    Load and return the 44-dim feature vector for patient `patient_id`.

    Args:
        patient_id : str or int, e.g. "1" or 1
        train      : if True, apply lightweight augmentation

    Returns:
        np.ndarray of shape (44,), dtype float32
    """
    pid = str(patient_id)
    upper_path = os.path.join(DATASET_ROOT, pid, "upper.stl")
    lower_path = os.path.join(DATASET_ROOT, pid, "lower.stl")

    if not os.path.exists(upper_path):
        raise FileNotFoundError(f"upper.stl not found for patient {pid}: {upper_path}")
    if not os.path.exists(lower_path):
        raise FileNotFoundError(f"lower.stl not found for patient {pid}: {lower_path}")

    features = extract_features(upper_path, lower_path)   # (44,)

    if train:
        features = _augment(features)

    return features


def _augment(features: np.ndarray) -> np.ndarray:
    """Apply lightweight augmentation to a feature vector."""
    rng = np.random.default_rng()

    # 1. Gaussian jitter ±2%
    noise = rng.normal(0.0, 0.02, size=features.shape).astype(np.float32)
    features = features * (1.0 + noise)

    # 2. Mirror augmentation (50% chance)
    if rng.random() < 0.5:
        features = features.copy()
        features[_FLIP_INDICES] *= -1.0

    return features
