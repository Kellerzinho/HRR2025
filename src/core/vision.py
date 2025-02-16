import torch
import cv2
import numpy as np

class VisionSystem:
    def __init__(self, model_path="models/best.pt"):
        # Carregar modelo YOLO (exemplo com ultralytics)
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        # ou com YOLOv8 (ultralytics>=8)
        # self.model = YOLO(model_path)

        # Configurações adicionais:
        self.conf_threshold = 0.5

    def detect_objects(self, frame):
        """
        Retorna lista de detecções com classe e bounding box.
        Classes esperadas: bola, robo_adversario, robo_aliado, trave, etc.
        """
        results = self.model(frame, size=640)
        detections = []
        for det in results.xyxy[0]:  # [x1, y1, x2, y2, conf, cls]
            x1, y1, x2, y2, conf, cls_id = det
            if conf < self.conf_threshold:
                continue
            label = results.names[int(cls_id)]
            detections.append({
                'label': label,
                'conf': float(conf),
                'bbox': (float(x1), float(y1), float(x2), float(y2))
            })
        return detections

    def segment_field(self, frame):
        """
        Opcional: Segmentar gramado usando HSV + thresholds de cor.
        Retornar máscara e contornos.
        """
        # Exemplo rápido
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Ajustar range para verde
        lower_green = np.array([30, 50, 50])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        return mask
