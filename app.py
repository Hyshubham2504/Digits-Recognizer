import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from streamlit_drawable_canvas import st_canvas

new_model = tf.keras.models.load_model('output_digits.h5')
st.title("Digit Recognizer")

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color='#ffffff',
    background_color='#000000',
    height=200,width=200
)

img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))

st.write(img.shape)

if st.button('Predict'):
	test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	op = new_model.predict(test.reshape(1,28,28))
	op = np.argmax(op)
	st.write(op)