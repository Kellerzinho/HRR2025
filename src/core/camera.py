import cv2
import yaml

class Camera:
    def __init__(self, calibration_file="config/camera_calibration.yaml", width=640, height=480, index=0):
        with open(calibration_file, 'r') as f:
            self.calib_data = yaml.safe_load(f)

        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.camera_matrix = self._parse_matrix(self.calib_data["camera_matrix"])
        self.dist_coeffs = self.calib_data["dist_coeffs"]
        self.offsets = self.calib_data["offsets"]  # offsets do torso/cabeça

    def _parse_matrix(self, mat_dict):
        # Converte dict para uma matriz real, se necessário
        # ...
        return None

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Se quiser fazer undistortion ou correções:
        # frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
        return frame

    def release(self):
        self.cap.release()
