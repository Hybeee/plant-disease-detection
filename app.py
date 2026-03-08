import streamlit as st
from ultralytics import YOLO
import PIL.Image

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
    st.image(res_plotted, caption='Detected Leaves', use_column_width=True)