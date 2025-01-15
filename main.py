import cv2
import numpy as np
from ultralytics import YOLO
import os
import requests

def download_model():
    url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
    model_path = "yolov5s.pt"
    if not os.path.exists(model_path):
        print("Downloading YOLOv5 model...")
        response = requests.get(url, stream=True)
        with open(model_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Model downloaded.")
    return model_path

class_names = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
    "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

def process_frame(frame, model):
    results = model.predict(source=frame, conf=0.5, device='cpu')
    detections = results[0].boxes.data.cpu().numpy()
    count = 0
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        if int(cls) in [2, 3, 5, 7]:
            count += 1
            class_name = class_names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, f'Total Vehicles: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    return frame


def process_video(source):
    cap = cv2.VideoCapture(source)
    model_path = download_model()
    model = YOLO(model_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, model)
        cv2.imshow('Vehicle Counting', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(source):
    frame = cv2.imread(source)
    model_path = download_model()
    model = YOLO(model_path)
    processed_frame = process_frame(frame, model)
    cv2.imshow('Vehicle Counting', processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def live_camera():
    process_video(0)

def main():
    while True:
        print("1. Upload Image")
        print("2. Upload Video")
        print("3. Live Camera")
        print("4. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            path = input("Enter image path: ")
            process_image(path)
        elif choice == '2':
            path = input("Enter video path: ")
            process_video(path)
        elif choice == '3':
            live_camera()
        elif choice == '4':
            break
        else:
            print("Invalid choice")

main()
