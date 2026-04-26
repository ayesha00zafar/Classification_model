# OccluzNet — AI Orthodontic Bite Image Classifier

Classifies dental images into 3 bite categories and generates orthodontic treatment plans.

| Class | Description |
|---|---|
| Normal / Class I | No significant malocclusion |
| Open Bite | Front teeth don't overlap vertically |
| Crossbite | Upper teeth bite inside lower teeth |

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Dataset Structure

Organise your images like this:

```
dataset/
    Normal_Class_I/
        img001.jpg
        img002.jpg
        ...
    Open_Bite/
        img001.jpg
        ...
    Crossbite/
        img001.jpg
        ...
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

## Train

```bash
python train_image.py --data dataset/ --epochs 20 --batch 16
```

Options:
- `--data`    Path to dataset root (default: `dataset/`)
- `--epochs`  Number of training epochs (default: `20`)
- `--batch`   Batch size (default: `16`)
- `--lr`      Learning rate (default: `0.0001`)
- `--dropout` Dropout rate (default: `0.4`)
- `--out`     Checkpoint output folder (default: `checkpoints/`)

Checkpoints are saved to `checkpoints/best_model.pt` and `checkpoints/final_model.pt`.

---

## Predict

Single image:
```bash
python predict.py --model checkpoints/best_model.pt --image patient_photo.jpg
```

With full treatment plan report:
```bash
python predict.py --model checkpoints/best_model.pt --image patient_photo.jpg --report
```

Batch prediction on a folder:
```bash
python predict.py --model checkpoints/best_model.pt --folder patient_images/ --report
```

---

## Sample Report Output

```
============================================================
  OCCLUZNET — AI ORTHODONTIC TREATMENT PLAN
============================================================
  Image    : patient_photo.jpg
  Diagnosis: Open Bite
  Confidence: 94.3%

  Recommendations
  ----------------------------------------
  1. Habit elimination therapy if tongue thrusting is present.
  2. Fixed orthodontic treatment with vertical elastics.
  3. Orthognathic surgery in severe skeletal cases.

  Estimated Treatment Duration: 18–30 months
  ⚠  Final diagnosis must be made by a licensed orthodontist.
============================================================
```

---

## Architecture

- **Backbone**: EfficientNet-B0 (pretrained on ImageNet)
- **Head**: Dropout → Linear(1280→512) → BN → ReLU → Dropout → Linear(512→3)
- **Loss**: CrossEntropyLoss with inverse-frequency class weights
- **Optimizer**: AdamW with cosine LR annealing
- **Augmentation**: Random crop, flip, colour jitter, rotation

---

## Files

| File | Purpose |
|---|---|
| `model.py` | EfficientNet-B0 model architecture |
| `dataset.py` | Dataset loader, augmentation, DataLoaders |
| `train_image.py` | Full training loop with per-class reporting |
| `predict.py` | Inference + treatment plan report generation |
| `requirements.txt` | Python dependencies |
