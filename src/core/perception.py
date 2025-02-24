import cv2
import torch
import math
import numpy as np
import yaml
from collections import deque
import time

class Perception:
    def __init__(
        self,
        calibration_file,
        camera_height=0.15,  # Exemplo de altura da câmera em relação ao chão
        roll=0.0,
        pitch=0.0,
        yaw=0.0,
        alpha=0.98
    ):
        """
        Inicializa a classe de percepção.
        :param calibration_file: Caminho para o arquivo YAML com parâmetros de calibração.
        :param camera_height: Altura da câmera em relação ao plano Z=0.
        :param roll, pitch, yaw: Ângulos em radianos representando a orientação do robô/câmera.
        """

        self.camera_matrix = None
        self.dist_coeffs = None

        # Carrega dados de calibração, se existirem
        self._load_calibration(calibration_file)

        self.camera_height = camera_height
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.mag_bias = np.array([0.0, 0.0, 0.0])

        # Buffer de dados do IMU (para sincronizar, se necessário)
        self.imu_data_queue = deque(maxlen=100)

        # Ganho do filtro complementar
        self.alpha = alpha
        # Timestamp para integração do IMU
        self.last_time = time.time()

        # Carrega o modelo MiDaS pequeno (DPT_Small) para depth estimation
        self.midas_type = "MiDaS_small"  # Modelo menor que outras variantes MiDaS
        print("[Perception] Carregando modelo MiDaS:", self.midas_type)
        self.midas = torch.hub.load("intel-isl/MiDaS", self.midas_type)
        self.midas.eval()

        # Carrega transforms recomendados para MiDaS
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.midas_transform = midas_transforms.dpt_transform

        # Define dispositivo (GPU se disponível, caso contrário CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.midas.to(self.device)
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(f"[Perception] Modelo de profundidade carregado em: {self.device}")

    def _load_calibration(self, calibration_file):
        """
        Lê o arquivo de calibração YAML e extrai os parâmetros intrínsecos
        e coeficientes de distorção da câmera.
        """
        try:
            with open(calibration_file, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            # Carrega matriz da câmera se existir
            cam_matrix_info = calib_data.get("camera_matrix", {})
            if cam_matrix_info and "data" in cam_matrix_info:
                data = cam_matrix_info["data"]
                rows = cam_matrix_info["rows"]
                cols = cam_matrix_info["cols"]
                self.camera_matrix = np.array(data).reshape((rows, cols)).astype(np.float32)
            else:
                print("[Perception] Aviso: arquivo de calibração não contém 'camera_matrix' válida.")
            
            # Carrega coeficientes de distorção
            dist = calib_data.get("dist_coeffs", None)
            if dist:
                self.dist_coeffs = np.array(dist).astype(np.float32)
            else:
                print("[Perception] Aviso: arquivo de calibração não contém 'dist_coeffs'.")

            if self.camera_matrix is not None and self.dist_coeffs is not None:
                print("[Perception] Calibração carregada com sucesso!")
            else:
                print("[Perception] Calibração incompleta ou ausente, seguindo sem correção.")
        except FileNotFoundError:
            print(f"[Perception] Arquivo de calibração não encontrado: {calibration_file}")
        except Exception as e:
            print(f"[Perception] Erro ao carregar calibração: {e}")

    def update_imu_data(self, imu_data):
        """
        Recebe leituras da IMU em tempo real.
        :param imu_data: dict com:
            {
                'timestamp': <float>,
                'accel': (ax, ay, az),
                'gyro': (gx, gy, gz),   # [rad/s]
                'mag': (mx, my, mz)    # campo magnético
            }
        """
            # Aplicar correção de hard-iron
        imu_data['mag'] = (imu_data['mag'][0] - self.mag_bias[0],
                        imu_data['mag'][1] - self.mag_bias[1],
                        imu_data['mag'][2] - self.mag_bias[2])
        self.imu_data_queue.append(imu_data)

    def fuse_orientation(self):
        """
        Faz a fusão dos dados mais recentes do giroscópio e do magnetômetro
        usando um Filtro Complementar (exemplo simples).
        """
        if not self.imu_data_queue:
            return  # sem dados, não faz nada

        # Pega a última leitura
        data = self.imu_data_queue[-1]
        now = data['timestamp']
        gx, gy, gz = data['gyro']
        ax, ay, az = data['accel']
        mx, my, mz = data['mag']

        dt = now - self.last_time if self.last_time is not None else 0.01
        self.last_time = now

        # Correção de pitch/roll via acelerômetro (filtro complementar)
        pitch_acc = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        roll_acc = math.atan2(ay, az)
        
        # Atualiza pitch e roll com filtro
        self.pitch = self.alpha * (self.pitch + gy * dt) + (1 - self.alpha) * pitch_acc
        self.roll = self.alpha * (self.roll + gx * dt) + (1 - self.alpha) * roll_acc

        # 1) Atualização via giroscópio (integração)
        #   yaw += gz * dt, pitch += gy * dt, roll += gx * dt (ex.: aproximando eixos)
        #   Observação: a ordem e a forma de integração dependem do frame da IMU
        self.yaw   += gz * dt
        self.pitch += gy * dt
        self.roll  += gx * dt

        # 2) Correção via magnetômetro (exemplo MUITO simplificado, assumindo roll e pitch pequenos)
        #   heading_mag = atan2(my, mx) -> dá um ângulo em rad
        heading_mag = math.atan2(my, mx)

        # Filtro Complementar para yaw:
        yaw_gyro = self.yaw
        yaw_mag = heading_mag
        # Ajustar diferenças de 2*pi, se necessário, para manter continuidade
        yaw_mag = self._normalize_angle(yaw_mag)
        yaw_gyro = self._normalize_angle(yaw_gyro)

        # Combina
        new_yaw = self.alpha * yaw_gyro + (1 - self.alpha) * yaw_mag

        # Salva resultado
        self.yaw = self._normalize_angle(new_yaw)
        # pitch e roll podem receber correções adicionais via acelerômetro se quisermos

    def _normalize_angle(self, angle):
        """
        Ajusta o ângulo para estar entre -pi e +pi, por exemplo.
        """
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def get_depth_map(self, frame_bgr):
        """
        Executa inferência no modelo MiDaS para obter o mapa de profundidade
        no tamanho da imagem original. Retorna depth_map como um array 2D.
        """
        # 1) Converte BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # 2) Aplica transform do MiDaS (normalização, resize etc.)
        input_batch = self.midas_transform(frame_rgb).to(self.device)

        # 3) Inferência do modelo
        with torch.no_grad():
            prediction = self.midas(input_batch)
            # Redimensiona para o tamanho original (altura, largura)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth_map = prediction.cpu().numpy()

        return depth_map

    def compute_world_coords(self, detection, depth=None):
        """
        Converte dados de detecção (ex.: centro da bola em pixels) para
        coordenadas no referencial do robô/campo, usando mapa de profundidade.

        :param detection: dict com infos como 'cx', 'cy', etc.
        :param depth_map: mapa de profundidade já calculado (opcional, mas preferível).
        :param frame_bgr: caso depth_map não seja passado, podemos gerar aqui (menos eficiente).
        :param method: string para definir qual método de profundidade usar (ex.: 'midas').
        :return: (X, Y, Z) em coordenadas do robô/campo, ou None se falhar.
        """
        cx = detection.get('cx', None)
        cy = detection.get('cy', None)
        if cx is None or cy is None:
            print("[Perception] Dados de detecção inválidos:", detection)
            return None

        if self.camera_matrix is None:
            print("[Perception] Sem matriz de câmera, não é possível calcular coords.")
            return None

        # Se não passamos depth_map, geramos agora (pode ser mais lento se houver muitas deteções)
        if depth is None:
            depth_map = depth
        else:
            print("[Perception] Método de profundidade desconhecido ou não implementado.")
            return None

        # Garante índices dentro da imagem
        h, w = depth_map.shape
        cx_int = min(max(int(cx), 0), w - 1)
        cy_int = min(max(int(cy), 0), h - 1)

        # Valor de profundidade no pixel desejado
        depth_value = depth_map[cy_int, cx_int]

        # Verifica se é válido
        if depth_value <= 0:
            print("[Perception] Profundidade inválida ou zero no ponto requisitado.")
            return None

        # Opcional: se a profundidade de MiDaS não for em escala métrica, aplicar fator
        # depth_value *= self.scale_factor  # Exemplo de calibração

        # 1) Undistort o ponto, para maior exatidão
        src_pts = np.array([[[cx_int, cy_int]]], dtype=np.float32)
        dst_pts = cv2.undistortPoints(src_pts, self.camera_matrix, self.dist_coeffs)
        u_nd, v_nd = dst_pts[0, 0]  # coords normalizadas

        # 2) Reprojetar em coords de câmera
        #    Xc = u_nd * Zc, Yc = v_nd * Zc, Zc = depth_value
        Xc = u_nd * depth_value
        Yc = v_nd * depth_value
        Zc = depth_value

        cam_point = np.array([Xc, Yc, Zc], dtype=np.float32)

        # 3) Rotação e Translação para coords do robô/campo
        R = self._euler_to_rotation_matrix(self.roll, self.pitch, self.yaw)
        T = np.array([0.0, 0.0, self.camera_height], dtype=np.float32)

        world_point = R @ cam_point + T

        X = world_point[0]
        Y = world_point[1]
        Z = world_point[2]

        return (X, Y)

    def process_detections(self, all_detections, depth=None):
        """
        Exemplo: itera sobre as detecções de 'ball', 'centercircle', 'penaltycross', etc.
        e converte cada uma em coordenadas no mundo usando o mapa de profundidade.

        :param all_detections: dict retornado pelo detector, ex.:
           {
             'ball': [ {...}, {...} ],
             'centercircle': [ {...}, ...],
             ...
           }
        :param frame_bgr: frame atual da câmera (BGR).
        :param method: método de profundidade (ex.: 'midas').
        :return: dicionário com coordenadas em (X, Y, Z), ex.:
           {
             'ball': [ (x1, y1, z1), (x2, y2, z2) ],
             'centercircle': [ (xc, yc, zc), ...],
             ...
           }
        """

        # Atualiza orientação (caso necessário, ex.: integrar dados de IMU)
        self.fuse_orientation()

        if depth is not None:
            depth_map = depth
        else:
            print("[Perception] Mapa de profundidade não fornecido, não é possível calcular coords.")
        

        # Dicionário para armazenar coordenadas no mundo
        world_positions = {
            'ball': [],
            'centercircle': [],
            'goal': [],
            'penaltycross': [],
            'line': [],
            'robot': []
        }

        # Itera sobre cada tipo de detecção
        for label, det_list in all_detections.items():
            for det in det_list:
                if label in ['ball', 'centercircle', 'penaltycross']:
                    pos = self.compute_world_coords(det, depth_map)
                    if pos is not None:
                        world_positions[label].append(pos)
                        if label == 'ball':
                            print("[Perception] Bola detectada em coords (X,Y,Z):", pos)

                elif label == 'goal':
                    # Exemplo: bounding box de um gol
                    bbox = det.get('bbox', None)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        temp_det = {'cx': cx, 'cy': cy}
                        pos = self.compute_world_coords(temp_det, depth_map)
                        if pos is not None:
                            world_positions[label].append(pos)

                elif label == 'line':
                    # Linhas do campo podem demandar lógica customizada (ex.: achar pontos específicos)
                    pass

                elif label == 'robot':
                    # Exemplo: outro robô detectado, pega base do bounding box
                    bbox = det.get('bbox', None)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2
                        cy = y2
                        temp_det = {'cx': cx, 'cy': cy}
                        pos = self.compute_world_coords(temp_det, depth_map)
                        if pos is not None:
                            world_positions[label].append(pos)

        return world_positions

    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        """
        Implementa a matriz de rotação com convenção Z-Y-X (yaw-pitch-roll).
        Ajuste caso sua convenção seja diferente.
        """
        R_x = np.array([
            [1, 0,           0          ],
            [0, math.cos(roll), -math.sin(roll)],
            [0, math.sin(roll),  math.cos(roll)]
        ])

        R_y = np.array([
            [ math.cos(pitch), 0, math.sin(pitch)],
            [ 0,               1, 0             ],
            [-math.sin(pitch), 0, math.cos(pitch)]
        ])

        R_z = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw),  math.cos(yaw), 0],
            [0,              0,             1]
        ])

        return R_z @ R_y @ R_x

