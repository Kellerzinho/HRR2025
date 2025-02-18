import time
import numpy as np
import yaml
from collections import deque
import math

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
                        if len(data) == rows * cols:
                            camera_matrix = np.array(data, dtype=np.float32).reshape((rows, cols))
                        else:
                            print("[Perception] Dados da camera_matrix com tamanho inesperado.")
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
        mx, my, mz = data['mag']

        dt = now - self.last_time if self.last_time is not None else 0.01
        self.last_time = now

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

    def compute_world_coords(self, detection, method='simple'):
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
            return None

        if self.camera_matrix is None:
            # Sem calibracao, não sabemos converter com precisão
            # Vamos retornar algo nulo ou um placeholder
            return None

        # Parâmetros da câmera
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx0 = self.camera_matrix[0, 2]  # principal point x
        cy0 = self.camera_matrix[1, 2]  # principal point y

        # Supondo que a bola está no chão e a câmera está a self.camera_height do chão
        # => Z = 0 (no referencial do campo), mas a câmera está a +camera_height
        # Esse approach assume um "pin-hole" e um triângulo simples:
        # Y (campo) = camera_height
        # focal = fx ou fy
        # (cx - cx0) => desloc. horizontal
        # (cy - cy0) => desloc. vertical
        # Distância é ~ camera_height / tan(theta)

        # Exemplo simplificado:
        # dZ = self.camera_height
        # Angulo vertical da bola:
        dy_pixels = (cy - cy0)
        # Se dy_pixels > 0, significa que está abaixo do centro da imagem
        # Razão pixel -> rad = arctan(dy_pixels / fy)

        if method == 'simple':
            # Tenta estimar a "distância na horizontal" e "lateral" assumindo um plano
            # e a altura self.camera_height
            # Esse é um mock: a forma exata requer geometria mais robusta
            angle_down = math.atan2(dy_pixels, fy)  # ângulo pra baixo em rad
            # Distância da câmera até o ponto no chão:
            dist_forward = self.camera_height / math.tan(abs(angle_down)) if angle_down != 0 else 999
            # Agora, o deslocamento horizontal (x) depende de (cx - cx0)
            dx_pixels = (cx - cx0)
            angle_side = math.atan2(dx_pixels, fx)
            dist_side = dist_forward * math.tan(angle_side)

            # No referencial da câmera (frente = +X, esquerda = +Y, se quiser)
            # Ou no ref do robô (frente = +X, esquerda = +Y, etc.)
            x_cam = dist_forward
            y_cam = dist_side

            # Ajuste usando yaw do robô
            # Se yaw = 0 significa que a frente do robô e do campo coincidem
            # Rotação 2D:
            cos_y = math.cos(self.yaw)
            sin_y = math.sin(self.yaw)

            # Transforma coords (x_cam, y_cam) do robo => do campo
            x_field = x_cam * cos_y - y_cam * sin_y
            y_field = x_cam * sin_y + y_cam * cos_y

            return (x_field, y_field)

        else:
            # Outros métodos (por exemplo, projetar contorno 3D)
            pass

        return None

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
                    pos = self.compute_world_coords(det, method='simple')
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
                        cy = (y1 + y2) / 2
                        temp_det = {'cx': cx, 'cy': cy}
                        pos = self.compute_world_coords(temp_det, method='simple')
                        if pos:
                            world_positions[label].append(pos)


        return world_positions
