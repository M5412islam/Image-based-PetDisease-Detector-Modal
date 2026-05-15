"""
Run this in FYP-Modal folder to find the correct confidence threshold.
Usage: python check_threshold.py

It tests every image in dataset/valid and shows you exactly what
threshold you should use in app.py and inference.py
"""

import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

MODEL_PATH       = "petcare_model3.keras"
CLASS_INDEX_PATH = "class_indices.json"
VAL_DIR          = "dataset/valid"
IMG_SIZE         = (224, 224)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH) as f:
    raw = json.load(f)
class_labels  = {v: k for k, v in raw.items()}
not_a_pet_idx = raw.get("Not_a_Pet", None)

# ── Collect confidence scores per class ───────────────────────────
results_by_class = {}   # class_name → list of max confidences
all_correct_confs = []  # confidence when prediction was CORRECT
all_wrong_confs   = []  # confidence when prediction was WRONG

print(f"Scanning {VAL_DIR}...\n")

for class_name in sorted(os.listdir(VAL_DIR)):
    class_dir = os.path.join(VAL_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    true_idx = raw.get(class_name, None)
    if true_idx is None:
        continue

    confs = []
    for fname in os.listdir(class_dir)[:50]:   # sample 50 per class
        fpath = os.path.join(class_dir, fname)
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        try:
            img = Image.open(fpath).convert("RGB").resize(IMG_SIZE)
            arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
            arr = preprocess_input(arr)
            probs   = model.predict(arr, verbose=0)[0]
            top_idx = int(np.argmax(probs))
            top_conf = float(probs[top_idx])
            confs.append(top_conf)
            if top_idx == true_idx:
                all_correct_confs.append(top_conf)
            else:
                all_wrong_confs.append(top_conf)
        except Exception:
            continue

    if confs:
        results_by_class[class_name] = confs

# ── Print per-class stats ──────────────────────────────────────────
print(f"{'Class':<45} {'Avg Conf':>9} {'Min Conf':>9} {'Max Conf':>9} {'Samples':>8}")
print("-" * 85)

for class_name in sorted(results_by_class.keys()):
    confs = results_by_class[class_name]
    flag = " ⚠" if np.mean(confs) < 0.45 else ""
    print(
        f"{class_name:<45}"
        f"{np.mean(confs)*100:>8.1f}%"
        f"{np.min(confs)*100:>8.1f}%"
        f"{np.max(confs)*100:>8.1f}%"
        f"{len(confs):>8}"
        f"{flag}"
    )

# ── Overall recommendation ────────────────────────────────────────
all_confs = [c for cl in results_by_class.values() for c in cl]
all_confs = np.array(all_confs)
all_correct_confs = np.array(all_correct_confs) if all_correct_confs else np.array([0])
all_wrong_confs   = np.array(all_wrong_confs)   if all_wrong_confs   else np.array([0])

print("\n" + "=" * 85)
print("OVERALL CONFIDENCE STATISTICS")
print("=" * 85)
print(f"  All predictions   — mean: {all_confs.mean()*100:.1f}%   "
      f"p10: {np.percentile(all_confs,10)*100:.1f}%   "
      f"p25: {np.percentile(all_confs,25)*100:.1f}%   "
      f"p50: {np.percentile(all_confs,50)*100:.1f}%")
print(f"  Correct only      — mean: {all_correct_confs.mean()*100:.1f}%   "
      f"p10: {np.percentile(all_correct_confs,10)*100:.1f}%")
print(f"  Wrong only        — mean: {all_wrong_confs.mean()*100:.1f}%   "
      f"p10: {np.percentile(all_wrong_confs,10)*100:.1f}%")

# Find threshold that keeps 85% of correct predictions
if len(all_correct_confs) > 0:
    suggested = float(np.percentile(all_correct_confs, 15))
else:
    suggested = 0.30

print(f"\n{'=' * 85}")
print(f"  RECOMMENDED THRESHOLD  →  {suggested*100:.1f}%  ({suggested:.3f})")
print(f"  (This keeps ~85% of correct predictions and rejects very low confidence ones)")
print(f"\n  ✅ Set this in app.py:      CONFIDENCE_THRESHOLD = {suggested:.2f}")
print(f"  ✅ Set this in inference.py: CONFIDENCE_THRESHOLD = {suggested:.2f}")
print(f"{'=' * 85}\n")

# Save to file
with open("confidence_stats.json", "w") as f:
    json.dump({
        "mean_confidence"        : float(all_confs.mean()),
        "p10_all"                : float(np.percentile(all_confs, 10)),
        "p25_all"                : float(np.percentile(all_confs, 25)),
        "mean_correct"           : float(all_correct_confs.mean()),
        "p10_correct"            : float(np.percentile(all_correct_confs, 10)),
        "suggested_threshold"    : suggested,
    }, f, indent=2)
print("Saved → confidence_stats.json")