import streamlit as st
import cv2
import pickle
import numpy as np
from ultralytics import YOLO
from collections import Counter
import io

gis_v8s = YOLO(r'runs/detect/train13/weights/best.pt')
gis_v8n = YOLO(r'runs/detect/train/weights/best.pt')

def predict_and_display(frame, model):
    if frame is None:
        st.write("Provided frame is not valid")
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640)
    annotated_image = results[0].plot()
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image_bgr

def anomaly_detection(frame, model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640)
    annotated_image = results[0].plot()
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image_bgr

####################################################################################################################################################################################################################################################################

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'mp4'])
model_choice = ['YOLO V8s', 'YOLO V8n']
selected_option = st.radio('Choose an option:', model_choice)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if uploaded_file.type.startswith('image'):
        frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
        if selected_option == 'YOLO V8s':
            annotated_frame = predict_and_display(frame, gis_v8s)
        elif selected_option == 'YOLO V8n':
            annotated_frame = predict_and_display(frame, gis_v8n)
        st.image(annotated_frame, channels='BGR', use_column_width=True)
    elif uploaded_file.type.startswith('video'):
        pass