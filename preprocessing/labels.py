"""
labels.py
---------
Parses Annotations.csv and returns structured label dicts per patient.

Label encoding
--------------
Right / Left Class  (5 classes):
  0  Class I
  1  Class II Edge to Edge
  2  Class II Full
  3  Class III
  4  Unknown

Anterior Bite  (4 classes):
  0  Normal
  1  Deep Bite
  2  Open Bite
  3  Inverted Bite
  -1 Unknown / missing  → patient will be skipped in training

Transversal Bite  (3 classes — coarsened from free-text):
  0  Normal
  1  Cross Bite  (any tooth spec)
  2  Scissor Bite

Median Lines  (2 classes):
  0  Centered
  1  Deviated
"""

import os
import pandas as pd

ANNOTATIONS_PATH = "dataset/Annotations.csv"

RIGHT_LEFT_MAP = {
    "Class I":               0,
    "Class II Edge to Edge": 1,
    "Class II Full":         2,
    "Class III":             3,
    "Unknown":               4,
}

ANTERIOR_MAP = {
    "Normal":        0,
    "Deep Bite":     1,
    "Open Bite":     2,
    "Inverted Bite": 3,
    "Unknown":      -1,   # triggers patient skip
}

MEDIAN_MAP = {
    "Centered": 0,
    "Deviated":  1,
}


def _encode_transversal(raw: str) -> int:
    """Coarsen free-text transversal field to 3 coarse classes."""
    r = str(raw).strip().lower()
    if r == "normal":
        return 0
    if "scissor" in r:
        return 2
    # "cross bite ...", "cross ..." → class 1
    return 1


# Cache so we only read the CSV once
_label_cache: dict | None = None


def _load_cache():
    global _label_cache
    if _label_cache is not None:
        return
    df = pd.read_csv(ANNOTATIONS_PATH)
    _label_cache = {}
    for _, row in df.iterrows():
        pid = str(int(row["Patient"]))
        _label_cache[pid] = {
            "right":       str(row["Right Class"]).strip(),
            "left":        str(row["Left Class"]).strip(),
            "anterior":    str(row["Anterior Bite"]).strip(),
            "transversal": str(row["Transversal Bite"]).strip(),
            "median":      str(row["Median Lines"]).strip(),
        }


def get_label(patient_id) -> dict | None:
    """
    Return raw string labels for a patient, or None if not found.

    Keys: right, left, anterior, transversal, median
    """
    _load_cache()
    return _label_cache.get(str(patient_id))


def encode_labels(label_dict: dict) -> dict | None:
    """
    Convert raw string label dict to integer encodings.

    Returns None if any critical label is invalid (triggers patient skip).
    Returns dict with keys: right, left, anterior, transversal, median
    """
    r = RIGHT_LEFT_MAP.get(label_dict["right"], 4)
    l = RIGHT_LEFT_MAP.get(label_dict["left"], 4)
    a = ANTERIOR_MAP.get(label_dict["anterior"], -1)
    t = _encode_transversal(label_dict["transversal"])
    m = MEDIAN_MAP.get(label_dict["median"], 1)

    if a == -1:
        return None   # skip patients with unknown anterior bite

    return {"right": r, "left": l, "anterior": a, "transversal": t, "median": m}
