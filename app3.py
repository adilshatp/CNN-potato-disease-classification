# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 15:06:02 2022

@author: ACER
"""


import streamlit as st
import tensorflow as tf
import streamlit as st
import pandas as pd

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "p_d",
    shuffle=True,
    image_size=(256,256),
    batch_size=32
)
class_names = dataset.class_names
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('potato_desease_detection.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # potato disease classification Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (256,256)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/256.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
   # predictions.rename(columns = {'0':'A','1':'b','2':'c'}, inplace = True)
    score = tf.nn.softmax(predictions[0])
    #st.write(predictions)
   #
    #st.write(score)
   # print(isinstance(score, str))
    st.write("This image most likely belongs to:",class_names[np.argmax(score)], 100 * np.max(score))       
    
   
    
    