"""
verify.py (fast version)
------------------------
Uses features_cache.npy instead of re-parsing STL files.
Near-instant for all 200 patients.

Usage:
    python verify.py --patient 1
    python verify.py --all
    python verify.py --val
"""

import os, sys, argparse, numpy as np, torch
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocessing.labels import get_label, encode_labels
from backend.model import OccluzNet

RIGHT_LEFT  = {0:"Class I", 1:"Class II E2E", 2:"Class II Full", 3:"Class III", 4:"Unknown"}
ANTERIOR    = {0:"Normal",  1:"Deep Bite", 2:"Open Bite", 3:"Inverted Bite"}
TRANSVERSAL = {0:"Normal",  1:"Cross Bite", 2:"Scissor Bite"}
MEDIAN      = {0:"Centered",1:"Deviated"}
DECODERS    = {"right": RIGHT_LEFT, "left": RIGHT_LEFT,
               "anterior": ANTERIOR, "transversal": TRANSVERSAL, "median": MEDIAN}
LABEL_KEYS  = ["right", "left", "anterior", "transversal", "median"]

def load_model():
    device = torch.device("cpu")
    model  = OccluzNet().to(device)
    if os.path.exists("backend/best_model.pt"):
        model.load_state_dict(torch.load("backend/best_model.pt", map_location=device))
    model.eval()
    return model, device

def load_cache():
    X   = np.load("dataset/features_cache.npy")
    Y   = np.load("dataset/labels_cache.npy")
    IDs = np.load("dataset/patient_ids.npy")
    return X, Y, IDs

def predict_all(model, device, X):
    with torch.no_grad():
        out = model(torch.tensor(X, dtype=torch.float32).to(device))
    return {k: torch.argmax(out[k], dim=1).numpy() for k in LABEL_KEYS}

def check_one(pid, X, Y, IDs, preds_all):
    pid_int = int(pid)
    idx = np.where(IDs == pid_int)[0]
    if len(idx) == 0:
        print(f"Patient {pid} not found in cache."); return
    i = idx[0]
    truth = {LABEL_KEYS[j]: int(Y[i, j]) for j in range(5)}
    preds = {k: int(preds_all[k][i]) for k in LABEL_KEYS}

    print(f"\n{'='*62}")
    print(f"  Patient {pid} — Prediction vs Ground Truth")
    print(f"{'='*62}")
    print(f"  {'Head':<14} {'Predicted':<22} {'Actual':<22} Match")
    print(f"  {'-'*58}")
    for k in LABEL_KEYS:
        p = DECODERS[k][preds[k]]
        t = DECODERS[k][truth[k]]
        m = "✓" if preds[k] == truth[k] else "✗"
        print(f"  {k:<14} {p:<22} {t:<22} {m}")
    print(f"{'='*62}\n")

def check_all(patient_ids, X, Y, IDs, preds_all):
    correct = {k: 0 for k in LABEL_KEYS}
    confusion = {k: {} for k in LABEL_KEYS}
    total = 0
    wrong = []

    for pid in patient_ids:
        pid_int = int(pid)
        idx = np.where(IDs == pid_int)[0]
        if len(idx) == 0:
            continue
        i = idx[0]
        total += 1
        all_ok = True
        for j, k in enumerate(LABEL_KEYS):
            pred  = int(preds_all[k][i])
            truth = int(Y[i, j])
            if pred == truth:
                correct[k] += 1
            else:
                all_ok = False
                key = (truth, pred)
                confusion[k][key] = confusion[k].get(key, 0) + 1
        if not all_ok:
            wrong.append(pid)

    print(f"\n{'='*52}")
    print(f"  Accuracy Report  ({total} patients)")
    print(f"{'='*52}")
    for k in LABEL_KEYS:
        acc = correct[k] / total * 100
        bar = "█" * int(acc / 5)
        print(f"  {k:<14} {acc:>5.1f}%  {bar}")
    mean = sum(correct[k]/total for k in LABEL_KEYS) / len(LABEL_KEYS) * 100
    print(f"  {'─'*46}")
    print(f"  {'Mean':<14} {mean:>5.1f}%")
    print(f"{'='*52}")

    # Show most common mistakes
    print(f"\n  Most common mistakes:")
    for k in LABEL_KEYS:
        if not confusion[k]:
            continue
        top = sorted(confusion[k].items(), key=lambda x: -x[1])[:3]
        dec = DECODERS[k]
        print(f"\n  {k}:")
        for (truth, pred), count in top:
            print(f"    Actual '{dec[truth]}' predicted as '{dec[pred]}' — {count}x")

    print(f"\n  Patients with ≥1 wrong: {len(wrong)}/{total}")

def get_val_ids(IDs):
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(IDs))
    split = int(0.8 * len(IDs))
    return IDs[idx[split:]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient", type=str, default=None)
    parser.add_argument("--all",  action="store_true")
    parser.add_argument("--val",  action="store_true")
    args = parser.parse_args()

    print("[INFO] Loading cache and model...")
    X, Y, IDs = load_cache()
    model, device = load_model()
    preds_all = predict_all(model, device, X)
    print(f"[INFO] Ready. {len(IDs)} patients in cache.\n")

    if args.patient:
        check_one(args.patient, X, Y, IDs, preds_all)

    elif args.all:
        check_all([str(i) for i in IDs], X, Y, IDs, preds_all)

    elif args.val:
        val_ids = get_val_ids(IDs)
        print(f"[INFO] Checking {len(val_ids)} validation patients")
        check_all([str(i) for i in val_ids], X, Y, IDs, preds_all)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python verify.py --patient 1")
        print("  python verify.py --val")
        print("  python verify.py --all")