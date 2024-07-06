import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

CFG_MODEL_PATH = "models/yolov5s.pt"
# End of Configurations

st.header("Object Detection using YOLOv5s")

@st.cache_resource
def loadmodel():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=CFG_MODEL_PATH, force_reload=True)
    return model

img = st.file_uploader("Upload an image.", type=['png','jpg'])

if img is not None:
    model = loadmodel()
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
