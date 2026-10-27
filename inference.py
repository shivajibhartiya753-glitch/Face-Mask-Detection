import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


print("[INFO] Loading mask detector model...")
model = load_model("mask_model.h5")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

labels = {0: "Incorrect Mask", 1: "With Mask", 2: "Without Mask"}
colors = {0: (0, 165, 255), 1: (0, 255, 0), 2: (0, 0, 255)} 

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: 
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    
    for (x, y, w, h) in faces:
        
        face_crop = frame[y:y+h, x:x+w]
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        face_crop = preprocess_input(face_crop)
        
        
        preds = model.predict(face_crop, verbose=0)[0]
        class_idx = np.argmax(preds)
        
        label = labels[class_idx]
        color = colors[class_idx]
        
        
        text = f"{label}: {preds[class_idx]*100:.2f}%"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
    cv2.imshow("Face Mask Detector", frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()