import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

st.header("Object Detection using YOLOv5s")

torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', source='github')

img = st.file_uploader("Upload an image.", type=['png','jpg'])

if img is not None:
    PIL_image = Image.open(img).convert("RGB")
    image = np.asarray(PIL_image)

    results = model(image)

    fig = plt.figure(figsize=(6,5))
    plt.imshow(np.squeeze(results.render()))
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    st.write(fig)
    st.write(results)
