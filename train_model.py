import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

# ========================
# SETTINGS
# ========================
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir   = "dataset/valid"
test_dir  = "dataset/test"

# ========================
# DATA GENERATORS
# ========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    shear_range=5,
    fill_mode='nearest'
)

val_gen  = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"Total classes (including Not_a_Pet): {num_classes}")
print(f"Classes found: {list(train_data.class_indices.keys())}")

# ========================
# CLASS WEIGHTS
# ========================
classes = np.unique(train_data.classes)
weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_data.classes
)

not_a_pet_idx = train_data.class_indices.get("Not_a_Pet", -1)
class_weights = {}
for i, w in enumerate(weights):
    if i == not_a_pet_idx:
        class_weights[i] = 1.0      # don't over-penalise Not_a_Pet
    else:
        class_weights[i] = min(w, 3.0)

print("Class Weights:", class_weights)

# ========================
# SAVE CLASS LABELS
# ========================
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f, indent=2)
print("class_indices.json saved")

# ========================
# MODEL
# ========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.summary()

# ========================
# LOSS
# ========================
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ========================
# PHASE 1 - COMPILE & TRAIN
# ========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(patience=3, factor=0.3, min_lr=1e-7, verbose=1),
    ModelCheckpoint("best_model.keras", save_best_only=True, verbose=1)
]

print("\nPhase 1 Training...\n")
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# ========================
# PHASE 2 - FINE TUNING
# ========================
print("\nFine-tuning...\n")

for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
    loss=loss_fn,
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks
)

# ========================
# EVALUATE
# ========================
loss, acc = model.evaluate(test_data)
print(f"\nFINAL TEST ACCURACY: {acc*100:.2f}%")

# ========================
# PER-CLASS ACCURACY
# ========================
print("\nPer-class accuracy on test set:")
y_true, y_pred = [], []
for batch_x, batch_y in test_data:
    preds = model.predict(batch_x, verbose=0)
    y_true.extend(np.argmax(batch_y, axis=1))
    y_pred.extend(np.argmax(preds,   axis=1))
    if len(y_true) >= test_data.n:
        break

y_true = np.array(y_true[:test_data.n])
y_pred = np.array(y_pred[:test_data.n])

idx_to_class = {v: k for k, v in train_data.class_indices.items()}
for class_idx in sorted(idx_to_class.keys()):
    mask = y_true == class_idx
    if mask.sum() == 0:
        continue
    class_acc = (y_pred[mask] == class_idx).mean()
    print(f"  {idx_to_class[class_idx]:<45} {class_acc*100:.1f}%")

# ========================
# SAVE CONFIDENCE STATS
# ========================
val_probs = []
for batch_x, _ in val_data:
    val_probs.extend(model.predict(batch_x, verbose=0).max(axis=1).tolist())
    if len(val_probs) >= val_data.n:
        break
val_probs = np.array(val_probs[:val_data.n])

stats = {
    "mean_max_confidence" : float(val_probs.mean()),
    "p10_confidence"      : float(np.percentile(val_probs, 10)),
    "p25_confidence"      : float(np.percentile(val_probs, 25)),
    "suggested_threshold" : float(np.percentile(val_probs, 15))
}
with open("confidence_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nConfidence stats saved to confidence_stats.json")
print(f"Suggested threshold for inference.py: {stats['suggested_threshold']:.2f}")

# ========================
# SAVE MODEL
# ========================
model.save("petcare_model3.keras")
print("\nTraining Complete! Model saved as petcare_model3.keras")