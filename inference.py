"""
PetCare AI - Inference
=======================
Use this in your app to get predictions.

Usage:
    from inference import predict
    result = predict("image.jpg", "dog")   # or "cat"
"""

import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# ─────────────────────────────────────────
# CONFIG — update threshold after retraining
# Open confidence_stats.json and copy the
# "suggested_threshold" value here
# ─────────────────────────────────────────
MODEL_PATH           = "petcare_model3.keras"
CLASS_INDEX_PATH     = "class_indices.json"
IMG_SIZE             = (224, 224)
CONFIDENCE_THRESHOLD = 0.4740686118602752   # update this after retraining

# ─────────────────────────────────────────
# LOAD MODEL (once, cached)
# ─────────────────────────────────────────
_model         = None
_class_labels  = None
_not_a_pet_idx = None

def _load():
    global _model, _class_labels, _not_a_pet_idx
    if _model is None:
        print("Loading model...")
        _model = tf.keras.models.load_model(MODEL_PATH)
    if _class_labels is None:
        with open(CLASS_INDEX_PATH) as f:
            raw = json.load(f)
        _class_labels  = {v: k for k, v in raw.items()}
        _not_a_pet_idx = raw.get("Not_a_Pet", None)

# ─────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────
def predict(image_path: str, selected_species: str) -> dict:
    """
    Parameters
    ----------
    image_path       : path to uploaded image
    selected_species : "cat" or "dog" — whatever user selected in your UI

    Returns
    -------
    {
      "status"     : "ok" or "rejected",
      "reason"     : rejection message  (only when rejected),
      "prediction" : disease name       (only when ok),
      "confidence" : float 0-1,
      "top3"       : list of (label, confidence)
    }
    """
    _load()

    selected_species = selected_species.lower().strip()
    if selected_species not in ("cat", "dog"):
        return _reject("Please select either Cat or Dog.", 0.0, [])

    # Open and preprocess image
    try:
        img = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    except Exception:
        return _reject("Could not open image. Please upload a valid JPG or PNG.", 0.0, [])

    arr       = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    arr       = preprocess_input(arr)
    probs     = _model.predict(arr, verbose=0)[0]
    top3_idxs = np.argsort(probs)[-3:][::-1]
    top3      = [(_class_labels[i], float(probs[i])) for i in top3_idxs]
    top_idx   = top3_idxs[0]
    top_label = _class_labels[top_idx]
    top_conf  = float(probs[top_idx])

    # ── LAYER 1: Not_a_Pet check ──────────
    if _not_a_pet_idx is not None:
        nap_conf = float(probs[_not_a_pet_idx])
        if top_idx == _not_a_pet_idx:
            return _reject(
                f"This doesn't look like a disease image ({nap_conf:.0%} confidence). "
                f"Please upload a clear photo of the affected skin area.",
                top_conf, top3
            )
        if nap_conf > 0.35:
            return _reject(
                "The image doesn't clearly show a skin condition. "
                "Please upload a closer, well-lit photo of the affected area.",
                top_conf, top3
            )

    # ── LAYER 2: Confidence threshold ─────
    if top_conf < CONFIDENCE_THRESHOLD:
        return _reject(
            f"Confidence too low ({top_conf:.0%}). "
            f"Please upload a clearer photo where the affected skin fills the frame.",
            top_conf, top3
        )

    # ── LAYER 3: Species lock ──────────────
    expected = selected_species.capitalize() + "-"   # "Cat-" or "Dog-"
    if not top_label.startswith(expected):
        # Try next best prediction that matches species
        for label, conf in top3[1:]:
            if label.startswith(expected) and conf >= CONFIDENCE_THRESHOLD:
                return _ok(label, conf, top3)
        return _reject(
            f"You selected {selected_species.capitalize()} but the image doesn't "
            f"match. Please check your selection or upload a clearer photo.",
            top_conf, top3
        )

    return _ok(top_label, top_conf, top3)


def _ok(label, confidence, top3):
    return {
        "status"     : "ok",
        "species"    : "cat" if label.startswith("Cat-") else "dog",
        "prediction" : label,
        "confidence" : confidence,
        "top3"       : top3
    }

def _reject(reason, confidence, top3):
    return {
        "status"     : "rejected",
        "reason"     : reason,
        "confidence" : confidence,
        "top3"       : top3
    }


# ─────────────────────────────────────────
# TEST FROM TERMINAL
# python inference.py my_image.jpg dog
# ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python inference.py <image_path> <cat|dog>")
        sys.exit(1)

    r = predict(sys.argv[1], sys.argv[2])
    print("\n" + "="*50)
    if r["status"] == "rejected":
        print(f"REJECTED: {r['reason']}")
    else:
        print(f"RESULT   : {r['prediction']}")
        print(f"SPECIES  : {r['species']}")
        print(f"CONFIDENCE: {r['confidence']:.1%}")
    print("\nTop 3:")
    for label, conf in r["top3"]:
        print(f"  {label:<45} {conf:.1%}")
    print("="*50)