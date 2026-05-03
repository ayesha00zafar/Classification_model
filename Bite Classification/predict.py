"""
predict.py
----------
Run inference on a new patient's STL files and generate a
structured orthodontic treatment plan report.

Usage:
    python predict.py --upper path/to/upper.stl --lower path/to/lower.stl
    python predict.py --upper path/to/upper.stl --lower path/to/lower.stl --json report.json
"""

import os
import sys
import argparse
import json
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.stl_features import extract_features
from backend.model import OccluzNet

# ------------------------------------------------------------------
# Label decoders
# ------------------------------------------------------------------
RIGHT_LEFT_LABELS = {0:"Class I", 1:"Class II Edge to Edge",
                     2:"Class II Full", 3:"Class III", 4:"Unknown"}
ANTERIOR_LABELS   = {0:"Normal", 1:"Deep Bite", 2:"Open Bite", 3:"Inverted Bite"}
TRANSVERSAL_LABELS= {0:"Normal", 1:"Cross Bite", 2:"Scissor Bite"}
MEDIAN_LABELS     = {0:"Centered", 1:"Deviated"}

# ------------------------------------------------------------------
# Treatment plan text generators
# ------------------------------------------------------------------
def _molar_plan(right, left):
    m = {
        (0,0): "No molar correction needed. Maintain Class I occlusion.",
        (1,1): "Bilateral Class II. Options: functional appliance (growing), maxillary molar distalization, or extraction-based treatment.",
        (2,2): "Bilateral Class II Full. Orthognathic evaluation recommended. Fixed functional appliance or surgical consult for severe cases.",
        (3,3): "Bilateral Class III. Reverse-pull headgear (growing) or surgical consult (adults).",
        (1,0): "Asymmetric Class II (right). Unilateral molar distalization or asymmetric extraction.",
        (0,1): "Asymmetric Class II (left). Unilateral molar distalization or asymmetric extraction.",
        (3,0): "Asymmetric Class III (right). Asymmetric mechanics or midline correction.",
        (0,3): "Asymmetric Class III (left). Asymmetric mechanics or midline correction.",
    }
    return m.get((right,left),
        f"Mixed molar relationship (R:{RIGHT_LEFT_LABELS[right]}, L:{RIGHT_LEFT_LABELS[left]}). Custom mechanics required.")

def _anterior_plan(ant):
    return {
        0: "Normal overbite. No anterior correction needed.",
        1: "Deep Bite. Intrude anteriors and/or extrude posteriors. Options: utility arch, bite plate, aligners with bite ramps.",
        2: "Open Bite. Extrude anteriors or intrude posteriors. Assess tongue habit. TADs for posterior intrusion in severe cases.",
        3: "Inverted Bite (anterior crossbite). Immediate correction required. Options: Z-spring, catlan appliance, composite bite-jumping.",
    }.get(ant, "Unknown anterior bite — clinical assessment required.")

def _transversal_plan(trans):
    return {
        0: "No transversal discrepancy. Normal arch width.",
        1: "Cross Bite. Expansion indicated. RPE for skeletal, quad helix for dental crossbite.",
        2: "Scissor Bite (buccal crossbite). Constriction mechanics or contraction arch on affected side.",
    }.get(trans, "Unknown transversal — clinical assessment required.")

def _median_plan(median):
    return {
        0: "Median lines centered. No correction needed.",
        1: "Median line deviation. Asymmetric mechanics or interarch elastics. Surgical consult if deviation >4mm.",
    }.get(median, "Unknown median status.")

def _severity(right, left, anterior, transversal, median):
    score = sum([
        2 if right in (2,3) else (1 if right==1 else 0),
        2 if left  in (2,3) else (1 if left ==1 else 0),
        2 if anterior in (2,3) else (1 if anterior==1 else 0),
        1 if transversal in (1,2) else 0,
        1 if median==1 else 0,
    ])
    return "Mild" if score<=1 else ("Moderate" if score<=4 else "Severe")

# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
def run_inference(upper_path, lower_path, model_path="backend/best_model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = OccluzNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[INFO] Loaded weights: {model_path}")
    else:
        print(f"[WARN] No weights at {model_path} — untrained model (train first!)")

    features = extract_features(upper_path, lower_path)
    X = torch.tensor(features[np.newaxis], dtype=torch.float32).to(device)
    preds = model.predict(X)

    r  = preds["right"].item()
    l  = preds["left"].item()
    a  = preds["anterior"].item()
    t  = preds["transversal"].item()
    m  = preds["median"].item()

    return {
        "classification": {
            "right_molar_class":   RIGHT_LEFT_LABELS[r],
            "left_molar_class":    RIGHT_LEFT_LABELS[l],
            "anterior_bite":       ANTERIOR_LABELS[a],
            "transversal_bite":    TRANSVERSAL_LABELS[t],
            "median_lines":        MEDIAN_LABELS[m],
            "overall_severity":    _severity(r,l,a,t,m),
        },
        "treatment_plan": {
            "molar_correction":       _molar_plan(r,l),
            "anterior_correction":    _anterior_plan(a),
            "transversal_correction": _transversal_plan(t),
            "median_correction":      _median_plan(m),
        },
        "disclaimer": (
            "AI-generated from 3D scan geometry only. "
            "All recommendations must be reviewed by a licensed orthodontist before clinical use."
        )
    }

def _print_plan(plan):
    print("\n" + "="*62)
    print("   OCCLUZNET — AI ORTHODONTIC TREATMENT PLAN")
    print("="*62)
    print("\n── CLASSIFICATION ──")
    for k,v in plan["classification"].items():
        print(f"  {k.replace('_',' ').title():<26} {v}")
    print("\n── TREATMENT RECOMMENDATIONS ──")
    for k,v in plan["treatment_plan"].items():
        print(f"\n  {k.replace('_',' ').title()}:")
        for sent in v.split(". "):
            if sent.strip():
                print(f"    • {sent.strip()}.")
    print(f"\n  ⚠  {plan['disclaimer']}")
    print("="*62 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upper", required=True)
    parser.add_argument("--lower", required=True)
    parser.add_argument("--model", default="backend/best_model.pt")
    parser.add_argument("--json",  default=None)
    args = parser.parse_args()

    plan = run_inference(args.upper, args.lower, args.model)
    _print_plan(plan)
    if args.json:
        with open(args.json,"w") as f:
            json.dump(plan, f, indent=2)
        print(f"[INFO] Report saved → {args.json}")
