import cv2
import numpy as np
import torch
from torchvision import transforms as tf
from model import DigitNN
import torch.nn.functional as F

model=DigitNN()
weights_path = 'weights.pth'
model.load_state_dict(torch.load(weights_path))
model.eval()

hsv_array = np.load('hsv_value.npy')
lrange = hsv_array[0]
urange = hsv_array[1]

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

kernel = np.ones((5, 5), np.int8)
canvas = np.zeros((720, 1280, 3))

def masker(input):
    mask = cv2.inRange(input, lrange, urange)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

noise_thresh = 80

while True:
    _, frame = cap.read()

    if canvas is not None:
        canvas = np.zeros_like(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = masker(hsv)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > noise_thresh:
        c = max(contours, key=cv2.contourArea)
        x1, y1, w, h = cv2.boundingRect(c)

        frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 4)
        canvas = cv2.rectangle(canvas, (x1, y1), (x1 + w, y1 + h), [0, 255, 0], 4)

        roi = frame[y1 - 30:y1 + h + 30, x1 - 30:x1 + w + 30]
        
        if not roi.size == 0:
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi = masker(roi)
            
            tran = tf.Compose([
    		tf.ToTensor(),
      		tf.Resize((28, 28)),
    		tf.Normalize((0.5,), (0.5,))
			])
            
            roi_tensor = tran(roi)
            output = model(roi_tensor.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            predicted_class = torch.argmax(output, dim=1).item()
            cv2.putText(frame, f'Prediction : {predicted_class}', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2)
            print(probabilities)


    canvas[mask > 0] = [255, 255, 255]
    

    stacked = np.hstack((canvas, frame))
    cv2.imshow('Camera Feed', cv2.resize(stacked, None, fx=0.6, fy=0.6))

    if cv2.waitKey(150) == 27:
        break

    if cv2.waitKey(1) & 0xFF == ord('c'):
        canvas = None