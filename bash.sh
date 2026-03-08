#!/bin/bash

pip uninstall -y opencv-python opencv-contrib-python numpy

pip install "numpy<2.0.0" opencv-python-headless torch torchvision

pip install -r requirements.txt

pip install ultralytics --no-deps

streamlit run app.py