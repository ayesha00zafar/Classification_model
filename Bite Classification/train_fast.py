"""
train_fast.py  (v2 — 32-dim features, best accuracy)
"""
import os, sys, numpy as np, torch, torch.nn as nn, torch.optim as optim
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from backend.model import OccluzNet

EPOCHS     = 60
LR         = 0.001
BATCH_SIZE = 16
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("backend", exist_ok=True)
print(f"[INFO] Device: {DEVICE}")

try:
    X   = np.load("dataset/features_cache.npy")
    Y   = np.load("dataset/labels_cache.npy")
    IDs = np.load("dataset/patient_ids.npy")
except FileNotFoundError:
    print("\n[ERROR] Cache not found. Run: python cache_features.py\n"); sys.exit(1)

print(f"[INFO] Loaded cache: {X.shape[0]} patients, {X.shape[1]} features each")

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))
X, Y, IDs = X[idx], Y[idx], IDs[idx]
split = int(0.8 * len(X))
X_tr, Y_tr = X[:split], Y[:split]
X_va, Y_va = X[split:], Y[split:]
print(f"[INFO] Train: {len(X_tr)} | Val: {len(X_va)}")

KEYS   = ["right", "left", "anterior", "transversal", "median"]
X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
X_va_t = torch.tensor(X_va, dtype=torch.float32)
Y_tr_t = {k: torch.tensor(Y_tr[:, i], dtype=torch.long) for i, k in enumerate(KEYS)}
Y_va_t = {k: torch.tensor(Y_va[:, i], dtype=torch.long) for i, k in enumerate(KEYS)}

# Flip indices for 32-dim v2 features
_FLIP_IDX = [0, 4, 6, 8, 16]

def augment_batch(X_batch):
    X_aug = X_batch.clone()
    noise = torch.randn_like(X_aug) * 0.02
    X_aug = X_aug * (1.0 + noise)
    for j in range(len(X_aug)):
        if torch.rand(1).item() < 0.5:
            X_aug[j, _FLIP_IDX] *= -1.0
    return X_aug

model = OccluzNet().to(DEVICE)

right_w       = torch.tensor([1.0, 1.7, 2.8, 4.3, 15.0]).to(DEVICE)
left_w        = torch.tensor([1.0, 1.7, 3.2, 3.2, 15.0]).to(DEVICE)
anterior_w    = torch.tensor([1.0, 1.1, 2.0, 22.0]).to(DEVICE)
transversal_w = torch.tensor([1.0, 3.5, 5.0]).to(DEVICE)
median_w      = torch.tensor([1.6, 1.0]).to(DEVICE)

criteria = {
    "right":       nn.CrossEntropyLoss(weight=right_w),
    "left":        nn.CrossEntropyLoss(weight=left_w),
    "anterior":    nn.CrossEntropyLoss(weight=anterior_w),
    "transversal": nn.CrossEntropyLoss(weight=transversal_w),
    "median":      nn.CrossEntropyLoss(weight=median_w),
}

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

def combined_loss(out, tgt):
    return sum(criteria[k](out[k], tgt[k].to(DEVICE)) for k in KEYS) / len(KEYS)

def per_acc(out, tgt):
    return {k: (torch.argmax(out[k],1)==tgt[k].to(DEVICE)).float().mean().item() for k in KEYS}

best_val_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    tr_loss, tr_acc, nb = 0.0, {k:0.0 for k in KEYS}, 0
    perm = torch.randperm(len(X_tr_t))

    for i in range(0, len(X_tr_t), BATCH_SIZE):
        bi  = perm[i:i+BATCH_SIZE]
        Xb  = augment_batch(X_tr_t[bi]).to(DEVICE)
        yb  = {k: Y_tr_t[k][bi] for k in KEYS}
        out = model(Xb)
        loss= combined_loss(out, yb)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tr_loss += loss.item()
        for k,v in per_acc(out, yb).items(): tr_acc[k] += v
        nb += 1

    tr_loss /= max(nb,1)
    for k in tr_acc: tr_acc[k] /= max(nb,1)

    model.eval()
    va_loss, va_acc, nv = 0.0, {k:0.0 for k in KEYS}, 0
    with torch.no_grad():
        for i in range(0, len(X_va_t), BATCH_SIZE):
            Xb  = X_va_t[i:i+BATCH_SIZE].to(DEVICE)
            yb  = {k: Y_va_t[k][i:i+BATCH_SIZE] for k in KEYS}
            out = model(Xb)
            va_loss += combined_loss(out, yb).item()
            for k,v in per_acc(out, yb).items(): va_acc[k] += v
            nv += 1
    va_loss /= max(nv,1)
    for k in va_acc: va_acc[k] /= max(nv,1)
    mean_va = sum(va_acc.values()) / len(va_acc)

    print(f"\nEpoch {epoch+1:>2}/{EPOCHS}  train={tr_loss:.4f}  val={va_loss:.4f}  val_acc={mean_va*100:.1f}%")
    for k in KEYS:
        bar = "█" * int(va_acc[k]*20)
        print(f"  {k:<14} train={tr_acc[k]*100:.1f}%  val={va_acc[k]*100:.1f}%  {bar}")

    scheduler.step(va_loss)

    if mean_va > best_val_acc:
        best_val_acc = mean_va
        torch.save(model.state_dict(), "backend/best_model.pt")
        print(f"  ✓ Saved best_model.pt  (val_acc={best_val_acc*100:.1f}%)")

torch.save(model.state_dict(), "backend/occluz_model.pt")
print(f"\n[DONE] Best val acc: {best_val_acc*100:.1f}%")