# aerial-object-classification-detection
Deep learning project for classifying aerial images as Bird or Drone using CNN, transfer learning, and optional YOLOv8 object detection with Streamlit deployment.
# 🛰️ Aerial Object Classification & Detection

A deep learning project that classifies aerial images as **Bird 🐦 or Drone 🚁** and optionally detects them using **YOLOv8 object detection**.

This system can be used in security surveillance, wildlife monitoring, airport safety, and airspace protection.

---

## 📌 Problem Statement

The goal of this project is to develop a deep learning-based solution that:

- Classifies aerial images into **Bird** or **Drone**
- Optionally performs **object detection** to locate objects in real-world scenes
- Deploys the final model using **Streamlit** for interactive use

Accurate identification between birds and drones is critical for:
- Security & defense surveillance
- Wildlife protection
- Airport bird-strike prevention
- Environmental research

---

## 🚀 Features

- ✅ Binary Image Classification (Bird vs Drone)
- ✅ Custom CNN Model
- ✅ Transfer Learning (ResNet50, MobileNet, EfficientNet)
- ✅ YOLOv8 Object Detection (Optional)
- ✅ Model Evaluation (Accuracy, Precision, Recall, F1-score)
- ✅ Confusion Matrix & Training Graphs
- ✅ Streamlit Web App Deployment

---

## 🧠 Tech Stack

- Python
- TensorFlow / Keras or PyTorch
- OpenCV
- NumPy & Pandas
- Matplotlib / Seaborn
- YOLOv8 (Ultralytics)
- Streamlit

---

## 📂 Dataset Information

### 📌 Classification Dataset
- Binary classes: **Bird / Drone**
- Image format: `.jpg`
- RGB images
- Pre-split into:
  - Train
  - Validation
  - Test

### 📌 Object Detection Dataset (YOLOv8 Format)
- 3319 images
- YOLO format annotation files (`.txt`)
- Bounding box format:
