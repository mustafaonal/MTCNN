import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN

mtcnn = MTCNN()

v_cap = cv2.VideoCapture('8-MaleNoGlasses.avi')

v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames = []

print(v_len)
for _ in range(v_len):
	ret, frame = v_cap.read()

	if not ret:
		continue

	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = mtcnn.detect(frame, landmarks=True)
	frames.append(frame)
	cv2.imshow('frame', frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


"""save_paths = [f'image_{i}.jpg' for i in range(len(frames))]

for frame, path in zip(frames, save_paths):
	mtcnn(frame, save_path=path)"""

cv2.imshow('frame', frames[100])
cv2.waitKey(0) #wait until any key pressed or cv2.waitKey(10) wait 110 sec or key pressed
cv2.destroyAllWindows()


