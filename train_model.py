import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

# =========================
# 📁 PATHS
# =========================
train_dir = "dataset/train"
val_dir = "dataset/valid"
test_dir = "dataset/test"

# =========================
# 🔁 DATA AUGMENTATION (IMPROVED)
# =========================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_gen = ImageDataGenerator(rescale=1./255)

# =========================
# 📊 LOAD DATA
# =========================
train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_data = val_test_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

test_data = val_test_gen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# =========================
# 🏷️ SAVE CLASS LABELS
# =========================
class_indices = train_data.class_indices
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f)

print("✅ Classes Saved:", class_indices)

# =========================
# ⚖️ CLASS WEIGHTS (IMBALANCE FIX)
# =========================
classes = train_data.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(classes),
    y=classes
)

class_weights = dict(enumerate(class_weights))
print("✅ Class Weights:", class_weights)

# =========================
# 🧠 MODEL BUILDING
# =========================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze initially

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_indices), activation='softmax')
])

# =========================
# ⚙️ COMPILE
# =========================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# ⏹️ CALLBACKS
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# =========================
# 🚀 INITIAL TRAINING
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# =========================
# 🔧 FINE-TUNING (UNFREEZE LAST 50 LAYERS)
# =========================
print("\n🔧 Fine-tuning started...\n")

base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=[early_stop, checkpoint]
)

# =========================
# 🧪 TEST EVALUATION
# =========================
test_loss, test_acc = model.evaluate(test_data)

print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# =========================
# 💾 SAVE FINAL MODEL
# =========================
model.save("petcare_model2.keras")

print("\n🎉 Model training complete and saved as petcare_model2.keras")