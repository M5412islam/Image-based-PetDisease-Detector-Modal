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
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dir = "dataset/train"
val_dir = "dataset/valid"
test_dir = "dataset/test"

# ========================
# DATA GENERATORS (FIXED)
# ========================
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,   # ✅ CRITICAL FIX
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1]
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
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

# ========================
# CLASS WEIGHTS (CAPPED)
# ========================
classes = np.unique(train_data.classes)

weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=train_data.classes
)

class_weights = {i: min(w, 3.0) for i, w in enumerate(weights)}
print("✅ Class Weights:", class_weights)

# ========================
# SAVE CLASS LABELS
# ========================
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)

# ========================
# MODEL
# ========================
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze most layers
base_model.trainable = True
for layer in base_model.layers[:-100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# ========================
# LOSS (LABEL SMOOTHING)
# ========================
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

# ========================
# COMPILE
# ========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=['accuracy']
)

# ========================
# CALLBACKS
# ========================
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(patience=3, factor=0.3),
    ModelCheckpoint("best_model.keras", save_best_only=True)
]

# ========================
# TRAIN (PHASE 1)
# ========================
print("\n🚀 Phase 1 Training...\n")

model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks
)

# ========================
# FINE-TUNING
# ========================
print("\n🔧 Fine-tuning...\n")

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
# TEST
# ========================
loss, acc = model.evaluate(test_data)
print(f"\n🔥 FINAL TEST ACCURACY: {acc*100:.2f}%")

model.save("petcare_model3.keras")

print("\n🎉 Training Complete!")