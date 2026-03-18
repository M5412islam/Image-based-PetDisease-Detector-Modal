# 🐾 PetCare AI – Disease Detection Model

## 📌 Overview

PetCare AI is a deep learning-based system designed to **identify diseases in cats and dogs using images**.
The model takes an input image of a pet and predicts the **disease class**, after which the application provides **relevant information such as symptoms, treatment, and precautions**.

This project is developed as part of a Final Year Project (FYP) and focuses on applying **Computer Vision and Deep Learning** in veterinary assistance.

---
## 🔗 Dataset Link

Access the dataset here:
👉 https://drive.google.com/file/d/1gccZR4Txm61n5gCddXEA3JPD0VaB6rv9/view?usp=drive_link


## 🧠 Model Details

* **Model Used:** MobileNetV2 (Transfer Learning)
* **Framework:** TensorFlow / Keras
* **Approach:**

  * Pretrained model (ImageNet)
  * Fine-tuned on custom dataset
* **Input Size:** 224 × 224 images
* **Output:** Multi-class classification (pet diseases)

---

## ⚙️ How the System Works

1. User uploads an image of a pet (cat or dog)
2. The model processes the image
3. Predicts the **disease class**
4. Backend maps the disease to:

   * Description
   * Symptoms
   * Treatment
   * Precautions
5. Results are displayed to the user

---

## 📂 Dataset Structure

The dataset is organized in a **classification-friendly format** with separate folders for training, validation, and testing.

```
dataset/
│
├── train/
│   ├── Cat/
│   │   ├── alopecia/
│   │   ├── dental_infection/
│   │   ├── ear_mites/
│   │   ├── eye_infection/
│   │   ├── flea_allergy/
│   │   ├── fungal_infection/
│   │   ├── healthy/
│   │   ├── miliary_dermatitis/
│   │   ├── ringworm/
│   │   └── scabies/
│   │
│   └── Dog/
│       ├── bacterial_dermatosis/
│       ├── demodicosis/
│       ├── dental_infection/
│       ├── eye_infection/
│       ├── flea_allergy/
│       ├── fungal_infection/
│       ├── healthy/
│       ├── hypersensitivity_dermatitis/
│       ├── mange/
│       ├── ringworm/
│       └── scabies/
│
├── valid/
│   ├── Cat/
│   └── Dog/
│
└── test/
    ├── Cat/
    └── Dog/
```

---

## 📊 Dataset Description

* Contains images of **cats and dogs with various diseases**
* Organized into **multiple disease classes**
* Includes:

  * Healthy animals
  * Skin infections
  * Parasitic diseases
  * Eye and dental conditions
* Dataset is split into:

  * **80% Training**
  * **10% Validation**
  * **10% Testing**

---

## 🚀 Training Process

1. Data preprocessing (rescaling + augmentation)
2. Load MobileNetV2 (pretrained)
3. Freeze base layers
4. Add custom classification layers
5. Train on dataset
6. Fine-tune entire model
7. Evaluate on test data

---

## 📈 Features

* Multi-class disease classification
* Supports both **cats and dogs**
* Lightweight model (suitable for deployment)
* Scalable for future disease additions

---

## ⚠️ Limitations

* Model only predicts disease class (not medical diagnosis)
* Treatment suggestions are **predefined (not AI-generated)**
* Accuracy depends on dataset quality and balance

---

## 🔮 Future Improvements

* Increase dataset size for better accuracy
* Add more disease categories
* Integrate real-time camera detection
* Deploy as a mobile/web application
* Use advanced models (EfficientNet, Vision Transformers)

---

## 👨‍💻 Author

Final Year Project – PetCare AI
Bachelor’s in Software Engineering

---

## 📢 Note

This system is designed for **educational purposes** and should not replace professional veterinary consultation.
