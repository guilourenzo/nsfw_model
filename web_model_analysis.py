from matplotlib.pyplot import xticks
import streamlit as st
import os
import numpy as np
import pandas as pd
from keras.preprocessing import image as k_image
from keras.models import load_model
from PIL import Image
import io
from tempfile import NamedTemporaryFile

st.set_option('deprecation.showfileUploaderEncoding', False)

model = load_model(r"C:\\Users\\guilo\\mba-tcc\\data\\modelo_v1.h5")
image_size = 299

@st.cache()
def predict_single_image(image_file):
    
    image_data = k_image.load_img(image_file, target_size=(image_size, image_size))

    # Convert the loaded image file to a numpy array
    image_array = k_image.img_to_array(image_data)
    image_array /= 255

    x_train = []
    x_train.append(image_array)
    x_test = np.array(x_train)

    predictions = model.predict(x_test)

    return round(predictions[0][0]*100, 2), round(predictions[0][1]*100, 2)


st.title("Upload + Classification Example")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
temp_file = NamedTemporaryFile(delete=False)

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)
    st.image(image_uploaded, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    temp_file.write(uploaded_file.getvalue())
    
    label = predict_single_image(temp_file.name)
  
    st.write(pd.DataFrame({
        'NOT SAFE': str(round(label[0], 2)) + '%',
        'SAFE': str(round(label[1], 2)) + '%',
    }, index=[0]))
