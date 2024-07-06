import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
from collections import Counter

st.header("Object Detection using YOLOv5s")

img = st.file_uploader("Upload an image.", type=['png','jpg'])

if img is not None:
    model = YOLO('yolov5su.pt')

    PIL_image = Image.open(img).convert("RGB")
    image = np.asarray(PIL_image)

    results = model.predict(image)

    for result in results:
        objects = np.array(result.boxes.cls).astype(int).tolist()

        result_array = result.plot()
        data = Image.fromarray(result_array)

        fig = plt.figure(figsize=(6,5))
        plt.imshow(data)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        st.write(fig)
    
    object_names = []

    for obj in objects:
        if obj in result.names:
            name = result.names[obj]
            object_names.append(name)
        
    unique_object_names = set(object_names)

    counted_objects = Counter(object_names)

    found = []

    for names in unique_object_names:
        found.append(' {} ({}) '.format(names, counted_objects[names]).upper())
    
    st.info("Objects Identified: " + ", ".join(found))
