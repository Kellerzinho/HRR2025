import cv2
import numpy as np
import yaml

# Carregar calibração
with open("config/camera_calibration.yaml") as f:
    data = yaml.safe_load(f)
mtx = np.array(data['camera_matrix'])
dist = np.array(data['dist_coeffs'])

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    
    undistorted = cv2.undistort(frame, mtx, dist)
    combined = np.hstack((frame, undistorted))
    
    cv2.imshow('Original vs Undistorted', combined)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()