import streamlit as st
from ultralytics import YOLO
import PIL.Image
import cv2

st.title("Plant Disease Detection")

@st.cache_resource
def load_yolo():
    return YOLO("model/best.pt")

model = load_yolo()

uploaded_file = st.file_uploader("Choose a picture...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption='Uploaded picture', use_column_width=True)
    
    results = model(image)

    res_plotted = results[0].plot()
    res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    st.image(res_plotted_rgb, caption='Detected Leaves', use_column_width=True)