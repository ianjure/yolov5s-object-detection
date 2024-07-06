import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import cv2

st.header("Object Detection using YOLOv5s")

model = torch.hub.load('yolov5', 'yolov5s', source='local')

img = st.file_uploader("Upload an image.", type=['png','jpg'])

if img is not None:
    PIL_image = Image.open(img).convert("RGB")
    image = np.asarray(PIL_image)
    #corrected_colors = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  

    results = model(image)

    fig = plt.figure(figsize=(6,5))
    plt.imshow(np.squeeze(results.render()))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    st.write(fig)
    st.write(results)
