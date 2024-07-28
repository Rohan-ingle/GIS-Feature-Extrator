# import streamlit as st
# import cv2
# import pickle
# import numpy as np
# from skimage.feature import hog
# from skimage import exposure
# from ultralytics import YOLO
# from collections import Counter
# import io

# gis_v8s = YOLO(r"runs\detect\train13\weights\best.pt")
# gis_v8n = YOLO(r"runs\detect\train\weights\best.pt")

# def predict_and_display(frame, model):
#     if frame is None:
#         st.write("Provided frame is not valid")
#         return None
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model.predict(source=frame_rgb, imgsz=640)
#     annotated_image = results[0].plot()
#     annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
#     return annotated_image_bgr

# def anomaly_detection(frame, model):
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model.predict(source=frame_rgb, imgsz=640)
#     annotated_image = results[0].plot()
#     annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
#     return annotated_image_bgr

# ####################################################################################################################################################################################################################################################################

# uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'mp4'])
# model_choice = ['YOLO V8s', 'YOLO V8n']
# selected_option = st.radio('Choose an option:', model_choice)

# if uploaded_file is not None:
#     file_bytes = uploaded_file.read()
#     if uploaded_file.type.startswith('image'):
#         frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
#         if selected_option == 'YOLO V8s':
#             annotated_frame = predict_and_display(frame, gis_v8s)
#         elif selected_option == 'YOLO V8n':
#             annotated_frame = predict_and_display(frame, gis_v8n)
#         st.image(annotated_frame, channels='BGR', use_column_width=True)
#     elif uploaded_file.type.startswith('video'):
#         pass

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import base64

# Load the models
gis_v8s = YOLO(r'runs/detect/train13/weights/best.pt')
gis_v8n = YOLO(r'runs/detect/train/weights/best.pt')

def predict_and_display(frame, model):
    if frame is None:
        st.write("Provided frame is not valid")
        return None, None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(source=frame_rgb, imgsz=640)
    annotated_image = results[0].plot()
    annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    return annotated_image_bgr, results[0].boxes

# Streamlit app layout
st.title("GIS Feature Extrator based on YOLOv8")

uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'mp4'])
model_choice = ['YOLO V8s', 'YOLO V8n']
selected_option = st.radio('Choose an option:', model_choice)

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    if uploaded_file.type.startswith('image'):
        frame = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
        if selected_option == 'YOLO V8s':
            annotated_frame, boxes = predict_and_display(frame, gis_v8s)
        elif selected_option == 'YOLO V8n':
            annotated_frame, boxes = predict_and_display(frame, gis_v8n)
        st.image(annotated_frame, channels='BGR', use_column_width=True)
        
        # Extract features from the detected objects
        features = []
        for box in boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            features.append([class_id, confidence, x1, y1, x2, y2])
        
        # Create a DataFrame for the features
        features_df = pd.DataFrame(features, columns=['Class ID', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])
        st.table(features_df)

        # Provide a download button for the CSV file
        csv = features_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="features.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)