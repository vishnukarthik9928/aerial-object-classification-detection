import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/bird_drone_model.h5")

img_path = "bird_test.jpg"

img = Image.open(img_path)
img = img.resize((224,224))

img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array,axis=0)

prediction = model.predict(img_array)[0][0]

print("Prediction score:", prediction)

if prediction > 0.5:
    print("Prediction: Drone")
else:
    print("Prediction: Bird")
