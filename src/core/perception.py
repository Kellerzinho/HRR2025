import time
import numpy as np
import yaml
from collections import deque
import math
import cv2

class Perception:
    """
    Classe responsável por (1) ler e fundir dados de IMU (giroscópio, magnetômetro)
    para estimar a orientação do robô e (2) projetar objetos detectados pelo YOLO
    em coordenadas do robô/campo.
    """

    def __init__(self, camera_matrix=None, dist_coeffs=None, camera_height=0.25, alpha=0.98, calibration_file="config/camera_calibration.yaml"):
        """
        :param camera_matrix: Matriz intrínseca da câmera (np.array 3x3) para projeção.
        :param dist_coeffs: Coeficientes de distorção (array).
        :param camera_height: Altura (em metros) da câmera em relação ao chão (para cálculo de distância).
        :param alpha: Coeficiente do Filtro Complementar (entre 0 e 1).
        :param calibration_file: Caminho para o arquivo YAML com parâmetros de calibração.
        """
        # Se os parâmetros de calibração não foram fornecidos, tenta carregá-los do arquivo
        if camera_matrix is None or dist_coeffs is None:
            try:
                with open(calibration_file, 'r') as f:
                    calib_data = yaml.safe_load(f)
                # Carrega a câmera matrix: pode estar em formato de dict ou lista
                cam_info = calib_data.get("camera_matrix", None)
                if cam_info is not None:
                    if isinstance(cam_info, dict):
                        data = cam_info.get("data", [])
                        rows = cam_info.get("rows", 3)
                        cols = cam_info.get("cols", 3)
                        camera_matrix = np.array(data, dtype=np.float32).reshape((rows, cols))

                    elif isinstance(cam_info, list):
                        # Se já for uma lista de listas
                        camera_matrix = np.array(cam_info, dtype=np.float32)
                else:
                    print("[Perception] camera_matrix não encontrada no arquivo de calibração.")

                # Carrega os coeficientes de distorção
                dcoeffs = calib_data.get("dist_coeffs", None)
                if dcoeffs is not None:
                    dist_coeffs = np.array(dcoeffs, dtype=np.float32)
                else:
                    print("[Perception] dist_coeffs não encontrados no arquivo de calibração.")

            except Exception as e:
                print("[Perception] Erro ao carregar calibração:", e)
                camera_matrix = None
                dist_coeffs = None

        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_height = camera_height
        self.mag_bias = np.array([0.0, 0.0, 0.0])

        # Buffer de dados do IMU (para sincronizar, se necessário)
        self.imu_data_queue = deque(maxlen=100)

        # Estado de orientação (yaw, pitch, roll) estimado
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0

        # Ganho do filtro complementar
        self.alpha = alpha
        # Timestamp para integração do IMU
        self.last_time = time.time()


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

    def compute_world_coords(self, detection, method='advanced'):
        """
        Converte dados de detecção (ex.: centro da bola em pixel) para coordenadas
        no referencial do robô ou do campo.

        :param detection: dict com infos como 'cx', 'cy', 'radius', etc., e possivelmente 'mask' ou 'contour'.
        :param method: 'simple' ou outro approach para cálculo de profundidade.
        :return: (X, Y) em coordenadas do robô/campo (depende do seu frame de referência).
        """

        # Exemplo: se for a bola (label='ball'), obtemos (cx, cy) do image space
        cx = detection.get('cx', None)
        cy = detection.get('cy', None)
        if cx is None or cy is None:
            print("[Perception] Dados de detecção inválidos:", detection)
            return None

        if self.camera_matrix is None:
            # Sem calibracao, não sabemos converter com precisão
            # Vamos retornar algo nulo ou um placeholder
            print("[Perception] Sem matriz de câmera, não é possível calcular coordenadas.")
            return None

        # Passo 1: Correção de distorção
        src_pts = np.array([[[cx, cy]]], dtype=np.float32)
        dst_pts = cv2.undistortPoints(src_pts, self.camera_matrix, self.dist_coeffs)
        u_norm, v_norm = dst_pts[0,0]
        
        # Passo 2: Vetor de direção normalizado
        x = (u_norm - self.camera_matrix[0,2]) / self.camera_matrix[0,0]
        y = (v_norm - self.camera_matrix[1,2]) / self.camera_matrix[1,1]
        dir_cam = np.array([x, y, 1.0])
        
        # Passo 3: Rotação considerando pitch/roll/yaw
        R = self._euler_to_rotation_matrix(self.roll, self.pitch, self.yaw)
        dir_world = R @ dir_cam
        
        # Passo 4: Interseção com o plano Z=0 (chão)
        if dir_world[2] == 0:
            return None
        t = -self.camera_height / dir_world[2]
        x = dir_world[0] * t
        y = dir_world[1] * t
        
        return (x, y)
    
    def _euler_to_rotation_matrix(self, roll, pitch, yaw):
        # Implementação correta da matriz de rotação ZYX
        R_x = np.array([[1, 0, 0],
                        [0, math.cos(roll), -math.sin(roll)],
                        [0, math.sin(roll), math.cos(roll)]])
        
        R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                        [0, 1, 0],
                        [-math.sin(pitch), 0, math.cos(pitch)]])
        
        R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                        [math.sin(yaw), math.cos(yaw), 0],
                        [0, 0, 1]])
        
        return R_z @ R_y @ R_x

    def process_detections(self, all_detections):
        """
        Exemplo: itera sobre as detecções de 'ball', 'centercircle', 'penaltycross', etc.
        e converte cada uma em coordenadas do mundo.
        
        :param all_detections: dict retornado por vision, ex.:
           {
             'ball': [ {...}, {...} ],
             'centercircle': [ {...}, ...],
             ...
           }
        :return: dicionário com coordenadas no campo ou no robô, ex.:
           {
             'ball': [ (x1, y1), (x2, y2) ],
             'centercircle': [ (xc, yc), ...],
             ...
           }
        """

        # Primeiro, atualizamos a fusão de orientação
        self.fuse_orientation()

        world_positions = {
            'ball': [],
            'centercircle': [],
            'goal': [],
            'penaltycross': [],
            'line': [],
            'robot': []
        }

        # Para cada tipo de landmark
        for label, det_list in all_detections.items():
            for det in det_list:
                if label in ['ball', 'centercircle', 'penaltycross']:
                    # Supondo que cada det tem (cx, cy)
                    pos = self.compute_world_coords(det)
                    if pos:
                        world_positions[label].append(pos)
                        if label == 'ball':
                            print("[Perception] Bola detectada em coordenadas:", pos)

                elif label == 'goal':
                    # Com "goal", muitas vezes é um contorno grande; se quisermos
                    # converter em um ponto (ex.: meio do contorno):
                    # Pega bounding box ou contorno e calcula um "centro" aproximado
                    bbox = det.get('bbox', None)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2)/2
                        cy = (y1 + y2)/2
                        temp_det = {'cx': cx, 'cy': cy}
                        pos = self.compute_world_coords(temp_det)
                        if pos:
                            world_positions[label].append(pos)

                elif label == 'line':
                    # "line" é mais complicado, pois envolve contornos extensos.  Em
                    # localization, é comum processar as linhas para achar interseções
                    # e então converter essas interseções. Aqui fica a critério do
                    # desenvolvedor. Exemplo:
                    pass

                elif label == 'robot':
                    bbox = det.get('bbox', None)
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cx = (x1 + x2) / 2
                        cy = y2
                        temp_det = {'cx': cx, 'cy': cy}
                        pos = self.compute_world_coords(temp_det)
                        if pos:
                            world_positions[label].append(pos)


        return world_positions