"""
train_image.py — Training script for OccluzNet bite image classifier

Usage:
    python train_image.py --data dataset/ --epochs 20 --batch 16

Dataset must be organised as:
    dataset/
        Normal_Class_I/  *.jpg / *.png
        Open_Bite/       *.jpg / *.png
        Crossbite/       *.jpg / *.png
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix

from model   import OccluzNetImageClassifier, CLASS_NAMES
from dataset import get_dataloaders


# ──────────────────────────────────────────
# Args
# ──────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data",    type=str,   default="dataset/",  help="Path to dataset root")
parser.add_argument("--epochs",  type=int,   default=20)
parser.add_argument("--batch",   type=int,   default=16)
parser.add_argument("--lr",      type=float, default=1e-4)
parser.add_argument("--dropout", type=float, default=0.4)
parser.add_argument("--out",     type=str,   default="checkpoints/")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")


# ──────────────────────────────────────────
# Data
# ──────────────────────────────────────────
train_loader, val_loader, class_weights = get_dataloaders(
    args.data, batch_size=args.batch
)
class_weights = class_weights.to(DEVICE)


# ──────────────────────────────────────────
# Model, loss, optimizer, scheduler
# ──────────────────────────────────────────
model     = OccluzNetImageClassifier(dropout=args.dropout).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


# ──────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    acc      = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    return avg_loss, acc, all_preds, all_labels


# ──────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────
best_val_acc = -1

for epoch in range(1, args.epochs + 1):
    train_loss, train_acc, _, _ = run_epoch(train_loader, train=True)
    val_loss,   val_acc,   val_preds, val_labels = run_epoch(val_loader, train=False)

    scheduler.step()

    print(f"\n[Epoch {epoch:02d}/{args.epochs}]")
    print(f"  Train — Loss: {train_loss:.4f}  Acc: {train_acc:.2f}%")
    print(f"  Val   — Loss: {val_loss:.4f}  Acc: {val_acc:.2f}%")
    print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = os.path.join(args.out, "best_model.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  [SAVED] Best model → {ckpt_path}  (val acc: {val_acc:.2f}%)")

    # Per-class report every 5 epochs
    if epoch % 5 == 0 or epoch == args.epochs:
        print("\n  Per-class report (validation):")
        print(classification_report(
            val_labels, val_preds,
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0,
        ))
        cm = confusion_matrix(val_labels, val_preds)
        print("  Confusion matrix:")
        print("  " + "  ".join(f"{n[:8]:>8}" for n in CLASS_NAMES))
        for i, row in enumerate(cm):
            print(f"  {CLASS_NAMES[i][:8]:>8}  " + "  ".join(f"{v:>8}" for v in row))

# Final save
final_path = os.path.join(args.out, "final_model.pt")
torch.save(model.state_dict(), final_path)
print(f"\n[DONE] Training complete. Best val acc: {best_val_acc:.2f}%")
print(f"       Final model → {final_path}")
