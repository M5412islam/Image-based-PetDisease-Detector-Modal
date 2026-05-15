"""
Downloads Not_a_Pet images using multiple sources to cover a full household scope:

CIFAR-100 (no internet after first download, ~170MB):
  Furniture          : bed, chair, couch, table, wardrobe
  Electrical devices : clock, keyboard, lamp, telephone, television
  Food containers    : bottle, bowl, can, cup, plate
  Trees / Plants     : willow, oak, maple, pine, palm, mushroom, poppy, rose, sunflower, tulip

Open Images v7 via FiftyOne (internet required, downloads on demand):
  Indoor rooms       : kitchen, living room, bedroom, bathroom
  Laptops / Screens  : laptop, computer monitor, desktop computer
  Lawn / Grass       : grass, lawn, meadow
  Land / Floor       : floor, pavement, soil, gravel, sand, tiles
  Trees (outdoors)   : tree, forest, palm tree, pine tree

Fallback – Describable Textures Dataset (DTD) via Keras:
  Floor / Ground textures: woven, grid, banded, striated, marbled, etc.

Usage:
    # Minimal (CIFAR-100 only, no extra installs):
    python download_not_a_pet.py --source cifar

    # Full (recommended, requires fiftyone):
    pip install fiftyone
    python download_not_a_pet.py --source all

    # Open Images only:
    python download_not_a_pet.py --source openimages
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
SPLITS = {
    "dataset/train/Not_a_Pet": 600,
    "dataset/valid/Not_a_Pet": 100,
    "dataset/test/Not_a_Pet" : 100,
}
TOTAL    = sum(SPLITS.values())   # 800
IMG_SIZE = (224, 224)

# ─────────────────────────────────────────
# CIFAR-100 SUPERCLASS CONFIG
# ─────────────────────────────────────────
# Superclass index → description
HOUSEHOLD_SUPERCLASSES = {
    3 : "Food containers    (bottle, bowl, can, cup, plate)",
    5 : "Electrical devices (clock, keyboard, lamp, telephone, television)",
    6 : "Furniture          (bed, chair, couch, table, wardrobe)",
    17: "Trees / Plants     (willow, oak, maple, pine, palm, mushroom, poppy, rose, sunflower, tulip)",
}

# ─────────────────────────────────────────
# OPEN IMAGES LABELS
# ─────────────────────────────────────────
OPEN_IMAGES_LABELS = [
    # Laptops / Computers
    "Laptop",
    "Computer monitor",
    "Desktop computer",
    # Indoor rooms (scene-level, use object labels as proxy)
    "Kitchen & dining room table",
    "Sofa bed",
    "Bed",
    "Bathroom cabinet",
    # Lawn / Grass
    "Grass",
    "Lawn",
    # Land / Floor textures
    "Floor",
    "Flooring",
    "Tile",
    "Pavement",
    # Trees / Outdoors
    "Tree",
    "Palm tree",
    "Christmas tree",
]

OPEN_IMAGES_PER_LABEL = 30   # images to fetch per OI label


# ═══════════════════════════════════════════
# HELPER: save numpy image array → JPEG
# ═══════════════════════════════════════════
def save_array(img_array: np.ndarray, path: str) -> None:
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img.save(path, "JPEG", quality=90)


def save_pil(pil_img: Image.Image, path: str) -> None:
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    pil_img.save(path, "JPEG", quality=90)


# ═══════════════════════════════════════════
# SOURCE 1 – CIFAR-100
# ═══════════════════════════════════════════
def load_cifar100_household() -> np.ndarray:
    import tensorflow as tf

    print("Loading CIFAR-100 (downloads once ~170 MB if not cached)…")
    (x_tr, _), (x_te, _) = tf.keras.datasets.cifar100.load_data(label_mode="fine")
    (_, y_tr_c), (_, y_te_c) = tf.keras.datasets.cifar100.load_data(label_mode="coarse")

    all_imgs   = np.concatenate([x_tr, x_te], axis=0)
    all_coarse = np.concatenate([y_tr_c, y_te_c], axis=0).flatten()

    sc_ids = list(HOUSEHOLD_SUPERCLASSES.keys())
    mask   = np.isin(all_coarse, sc_ids)
    imgs   = all_imgs[mask]
    lbls   = all_coarse[mask]

    print("  CIFAR-100 household images per superclass:")
    for sc, desc in HOUSEHOLD_SUPERCLASSES.items():
        print(f"    [{sc:2d}] {desc}  →  {(lbls == sc).sum()} images")
    print(f"  CIFAR-100 total: {len(imgs)}")
    return imgs


# ═══════════════════════════════════════════
# SOURCE 2 – OPEN IMAGES v7 via FiftyOne
# ═══════════════════════════════════════════
def load_open_images(tmp_dir: str = "/tmp/oi_not_a_pet") -> list[np.ndarray]:
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("  [OpenImages] fiftyone not installed – skipping.")
        print("  Install with:  pip install fiftyone")
        return []

    os.makedirs(tmp_dir, exist_ok=True)
    arrays: list[np.ndarray] = []

    for label in OPEN_IMAGES_LABELS:
        print(f"  [OpenImages] fetching '{label}' …", end=" ", flush=True)
        try:
            ds = foz.load_zoo_dataset(
                "open-images-v7",
                split="validation",
                label_types=["detections"],
                classes=[label],
                max_samples=OPEN_IMAGES_PER_LABEL,
                dataset_name=f"oi_{label.replace(' ', '_').lower()}",
                dataset_dir=os.path.join(tmp_dir, label.replace(" ", "_")),
            )
            for sample in ds:
                try:
                    img = Image.open(sample.filepath)
                    arrays.append(np.array(img.convert("RGB")))
                except Exception:
                    pass
            print(f"{len(ds)} images")
            fo.delete_dataset(ds.name)
        except Exception as e:
            print(f"SKIPPED ({e})")

    print(f"  OpenImages total: {len(arrays)}")
    return arrays


# ═══════════════════════════════════════════
# SOURCE 3 – DTD floor/texture fallback
# ═══════════════════════════════════════════
def load_dtd_textures() -> np.ndarray:
    """
    DTD (Describable Textures Dataset) ships with TF Datasets.
    We take a broad sample as stand-in for floor / ground / wall textures.
    """
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("  [DTD] tensorflow-datasets not installed – skipping.")
        print("  Install with:  pip install tensorflow-datasets")
        return np.empty((0, 32, 32, 3), dtype=np.uint8)

    print("  [DTD] loading texture dataset…")
    ds = tfds.load("dtd", split="train", as_supervised=False)
    arrays = []
    for ex in ds.take(300):
        img = ex["image"].numpy()
        arrays.append(img)
    print(f"  DTD total: {len(arrays)}")
    return np.array(arrays, dtype=object)   # variable sizes – handle at save time


# ═══════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["cifar", "openimages", "dtd", "all"],
        default="all",
        help="Which image source(s) to use (default: all)",
    )
    args = parser.parse_args()

    # ── collect images from each source ──────────────────────────────────────
    pool_arrays: list[np.ndarray] = []   # fixed-size uint8 arrays (H,W,3)
    pool_pil:    list[Image.Image] = []  # variable-size PIL images (e.g. DTD)

    if args.source in ("cifar", "all"):
        cifar_imgs = load_cifar100_household()
        pool_arrays.extend(cifar_imgs)

    if args.source in ("openimages", "all"):
        oi_imgs = load_open_images()
        pool_pil.extend([Image.fromarray(a) for a in oi_imgs])

    if args.source in ("dtd", "all"):
        dtd_imgs = load_dtd_textures()
        pool_pil.extend([Image.fromarray(a.astype(np.uint8)) for a in dtd_imgs])

    # ── merge into one PIL list ───────────────────────────────────────────────
    all_pil = [Image.fromarray(a) for a in pool_arrays] + pool_pil
    print(f"\nTotal images in pool : {len(all_pil)}")

    if len(all_pil) == 0:
        sys.exit("ERROR: no images collected. Run with --source cifar as a minimum.")

    # Shuffle
    np.random.shuffle(all_pil)

    total_needed = sum(SPLITS.values())
    if len(all_pil) < total_needed:
        print(
            f"Warning: only {len(all_pil)} images available "
            f"(need {total_needed}). Will repeat with wrap-around."
        )
        # Tile so we always have enough
        reps     = (total_needed // len(all_pil)) + 1
        all_pil  = (all_pil * reps)[:total_needed]

    # ── save into dataset folders ─────────────────────────────────────────────
    print()
    offset = 0
    for dest_dir, count in SPLITS.items():
        os.makedirs(dest_dir, exist_ok=True)
        batch = all_pil[offset: offset + count]
        for i, pil_img in enumerate(batch):
            out_path = os.path.join(dest_dir, f"not_a_pet_{offset + i:04d}.jpg")
            save_pil(pil_img, out_path)
        print(f"  Saved {len(batch)} images  →  {dest_dir}")
        offset += count

    print(f"\nDone!  Not_a_Pet folders are ready.")
    print("Run next:  python train_model.py")


if __name__ == "__main__":
    main()