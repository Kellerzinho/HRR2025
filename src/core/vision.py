import cv2
import numpy as np
from ultralytics import YOLO


class VisionSystem:
    def __init__(self, model_path="models/best.pt", conf_threshold=0.5):
        """
        Classe para executar detecção/segmentação de landmarks (ball, centercircle, goal, line, penaltycross, robot).
        
        model_path: Caminho para o modelo YOLOv8 treinado em segmentação.
        conf_threshold: Limite de confiança para filtrar detecções.
        """
        # Carrega o modelo YOLOv8
        try:
            self.model = YOLO(model_path)
            print(f"[Vision] Modelo YOLOv8 carregado: {model_path}")
        except Exception as e:
            print(f"[Vision] Erro ao carregar o modelo {model_path}: {e}")
            self.model = None

        self.conf_threshold = conf_threshold
        # Em alguns modelos, as classes podem ser setadas em self.model.names.
        # Certifique-se de que correspondem a:
        # 0=ball, 1=centercircle, 2=goal, 3=line, 4=penaltycross, 5=robot, etc.

    def detect_landmarks(self, frame):
        """
        Executa a inferência de segmentação no frame, retornando um dicionário com as informações
        de cada landmark. Exemplo de retorno:
        
        {
          'ball': [ {'conf': 0.95, 'bbox': (...), 'mask': (H,W) binária, 'cx': X, 'cy': Y, 'radius': R}, ... ],
          'centercircle': [ {'conf': 0.90, 'center': (cx, cy), 'radius': r, 'contour': [array de pontos] }, ... ],
          'goal': [ {'conf': 0.88, 'contour': [array], ... }, ... ],
          'line': [ {'conf': 0.92, 'contour': [array], ... }, ... ],
          'penaltycross': [ {...}, ...],
          'robot': [ {...}, ...]
        }
        """
        if self.model is None:
            return {}

        # Realiza a predição
        results = self.model.predict(source=frame, conf=self.conf_threshold, task='segment')  
        # O 'task=segment' deve assegurar que rode segmentação. Verifique se é necessário.

        # Normalmente, 'results' é uma lista, vamos pegar o primeiro (caso batch_size=1):
        if not results:
            return {}

        # results[0] contém as previsões para este frame
        r = results[0]
        boxes = r.boxes  # bounding boxes
        masks = r.masks  # máscaras de segmentação

        # Mapeia a classe -> lista de detections
        detections = {
            'ball': [],
            'centercircle': [],
            'goal': [],
            'line': [],
            'penaltycross': [],
            'robot': []
        }

        if boxes is None or masks is None:
            return detections

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())  # classe
            conf = float(box.conf[0].item()) # confiança
            label = r.names[cls_id] if r.names else str(cls_id)

            if label not in detections:
                # Se o modelo tiver classes adicionais, ignore.
                continue

            # Extraímos a máscara segmentada correspondente
            mask_i = masks.data[i]  # objeto Mask
            # Converte para array numpy binário
            mask_np = mask_i.cpu().numpy()  # shape: [H, W], valores [0,1]
            mask_bin = (mask_np * 255).astype(np.uint8)

            # Bounding box (x1, y1, x2, y2)
            x1, y1, x2, y2 = box.xyxy[0]
            bbox = (float(x1), float(y1), float(x2), float(y2))

            # Monta info básica
            det_info = {
                'conf': conf,
                'bbox': bbox,
                'mask': mask_bin
            }

            # Para cada tipo de landmark, podemos extrair parâmetros específicos
            if label == 'ball':
                # Exemplo: extrair centro e raio da bola (minEnclosingCircle)
                cx, cy, radius = self._extract_circle_params(mask_bin)
                det_info['cx'] = cx
                det_info['cy'] = cy
                det_info['radius'] = radius

            elif label == 'centercircle':
                # O círculo central deve ser um contorno amplo no meio do campo
                cx, cy, radius = self._extract_circle_params(mask_bin)
                det_info['center'] = (cx, cy)
                det_info['radius'] = radius

            elif label == 'penaltycross':
                # Geralmente um ponto pequeno ou mini círculo
                px, py, r_small = self._extract_circle_params(mask_bin)
                det_info['center'] = (px, py)
                det_info['radius'] = r_small

            elif label == 'goal':
                # Pode ser um retângulo grande ou contorno irregular
                # Extraímos contorno para posterior uso no SLAM
                cont = self._extract_contour(mask_bin)
                det_info['contour'] = cont

            elif label == 'line':
                # A linha pode ser muito extensa; podemos guardar o contorno
                cont = self._extract_contour(mask_bin)
                det_info['contour'] = cont

            elif label == 'robot':
                # Se quiser rastrear outros robôs, extrair contorno ou bounding box
                cont = self._extract_contour(mask_bin)
                det_info['contour'] = cont

            detections[label].append(det_info)

        return detections

    def _extract_contour(self, mask_bin):
        """
        Dado um array binário (0/255), encontra o contorno principal.
        Retorna a lista de pontos do maior contorno (pelo area).
        """
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        # Pega o maior contorno (em área)
        largest_cont = max(contours, key=cv2.contourArea)
        return largest_cont.squeeze()  # remove dimensão extra, se houver

    def _extract_circle_params(self, mask_bin):
        """
        Usar minEnclosingCircle para encontrar (cx, cy, radius) do contorno principal.
        Se não encontrar contorno, retorna (0,0,0).
        """
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, 0)
        largest_cont = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_cont)
        return (int(cx), int(cy), float(radius))

    def draw_segmentations(self, frame, detections):
        """
        Desenha as segmentações/contornos na imagem, útil para debug.
        Pode exibir bounding boxes ou contornos conforme o label.
        """
        color_map = {
            'ball': (0, 0, 255),           # Vermelho 
            'centercircle': (128, 0, 128), # Roxo 
            'goal': (255, 0, 0),           # Azul 
            'line': (255, 255, 0),         # Ciano 
            'penaltycross': (255, 0, 255), # Magenta 
            'robot': (0, 255, 255)         # Amarelo
        }

        for label, det_list in detections.items():
            for det in det_list:
                c = color_map.get(label, (255, 255, 255))
                # Se tivermos contorno
                if 'contour' in det and det['contour'] is not None:
                    cont = det['contour']
                    if len(cont.shape) < 3:
                        cont = cont.reshape(-1, 1, 2)
                    cv2.drawContours(frame, [cont], -1, c, 2)

                # Se for bola, circle center, penalty cross, etc.
                if label in ['ball', 'centercircle', 'penaltycross']:
                    cx = det.get('cx') or (det.get('center')[0] if 'center' in det else None)
                    cy = det.get('cy') or (det.get('center')[1] if 'center' in det else None)
                    r  = det.get('radius', 0)
                    if cx is not None and cy is not None:
                        cv2.circle(frame, (int(cx), int(cy)), int(r), c, 2)

                # Podemos também desenhar bounding box
                bbox = det.get('bbox', None)
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                    # Exibe label
                    conf = det['conf']
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

        return frame
