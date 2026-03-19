PetCare AI – Disease Detection Model
🔍 Overview

PetCare AI is a deep learning-based system designed to detect diseases in cats and dogs from images.
The model analyzes an uploaded pet image and predicts the Top-2 most likely diseases along with confidence scores, followed by relevant medical information.

This project is developed as a Final Year Project (FYP) and demonstrates the application of Computer Vision and Deep Learning in veterinary assistance.

📁 Dataset

👉 Dataset Link:
https://drive.google.com/file/d/1gccZR4Txm61n5gCddXEA3JPD0VaB6rv9/view?usp=drive_link

📊 Dataset Characteristics

Total Classes: 21 (Cat + Dog diseases)

Includes:

Skin diseases

Parasitic infections

Eye & dental conditions

Healthy cases

Dataset split:

80% Training

10% Validation

10% Testing

🧠 Model Architecture
🔹 Base Model (Transfer Learning)

MobileNetV2

Pretrained on ImageNet

Used as a feature extractor

🔹 Why MobileNetV2?

Lightweight and fast

Suitable for mobile deployment (React Native)

Good performance on image classification tasks

⚙️ Training Strategy

The model was trained in two phases:

🚀 Phase 1: Feature Extraction

Loaded MobileNetV2 (pretrained weights)

Frozen all base layers

Added custom classification head:

GlobalAveragePooling

Dense (128 neurons, ReLU)

Dropout (0.5)

Output layer (21 classes, Softmax)

Trained only the top layers

🔧 Phase 2: Fine-Tuning

Unfroze top layers of MobileNetV2

Fine-tuned deeper layers to adapt to dataset

Used low learning rate to avoid destroying pretrained features

🧪 Techniques Used
✔️ Data Preprocessing

Image resizing (224 × 224)

Normalization

✔️ Data Augmentation (Training only)

Rotation

Zoom

Horizontal flip

Brightness adjustment

✔️ Class Imbalance Handling

Dataset had imbalance (some classes <100 images)

Solved using:

Class Weights

Slight augmentation for minority classes

✔️ Optimization Techniques

Optimizer: Adam

Loss Function: Categorical Crossentropy

Callbacks Used:

EarlyStopping

ReduceLROnPlateau

ModelCheckpoint

📊 Model Performance
Metric	Value
Training Accuracy: ~94%
Validation Accuracy: ~78%
Test Accuracy: 74.56%

⚠️ Final performance depends on real-world unseen images.

🔄 System Workflow
User uploads image
        ↓
Model processes image
        ↓
Predicts Top-2 diseases
        ↓
Fetch disease data from backend
        ↓
Display:
  - Disease name
  - Confidence %
  - Description
  - Treatment
