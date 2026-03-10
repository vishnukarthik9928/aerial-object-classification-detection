import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Starting evaluation...")

# Test dataset directory
test_dir = "test"

# Image preprocessing
test_gen = ImageDataGenerator(rescale=1./255)

# Load test data
test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print("Test dataset loaded!")

# Load trained model
model = tf.keras.models.load_model("models/bird_drone_model.h5")

print("Model loaded successfully!")

# Make predictions
predictions = model.predict(test_data)
pred_classes = (predictions > 0.5).astype(int)

# Print results
print("\nClassification Report:")
print(classification_report(test_data.classes, pred_classes))

# Confusion Matrix
cm = confusion_matrix(test_data.classes, pred_classes)

print("\nConfusion Matrix:")
print(cm)

# -------------------------
# Plot Confusion Matrix
# -------------------------

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Bird","Drone"],
            yticklabels=["Bird","Drone"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# -------------------------
# Prediction Distribution Graph
# -------------------------

plt.figure(figsize=(6,4))
plt.hist(predictions, bins=20)
plt.title("Prediction Score Distribution")
plt.xlabel("Prediction Score")
plt.ylabel("Frequency")
plt.show()