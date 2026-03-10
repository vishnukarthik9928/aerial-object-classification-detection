import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/bird_drone_model.h5")

st.title("Aerial Object Classification")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img)

    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.success("Prediction: Drone")
    else:
        st.success("Prediction: Bird")
