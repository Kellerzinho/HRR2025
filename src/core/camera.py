import cv2
import yaml
import numpy as np

class Camera:
    def __init__(self, calibration_file, width=640, height=480, index=0):
        """
        Inicializa a câmera, carregando parâmetros de calibração (se existirem).
        
        :param calibration_file: Caminho para o arquivo YAML com parâmetros de calibração
        :param width: Largura desejada do frame (pode ser ajustado pela câmera se suportado)
        :param height: Altura desejada do frame
        :param index: Índice da câmera (0, 1, etc.); no Jetson Nano, normalmente 0 para uma webcam USB
        """
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None

        # Carrega dados de calibração, se existirem
        self._load_calibration(calibration_file)

        # Inicializa a captura
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = width
        self.height = height

        if not self.cap.isOpened():
            print(f"[Camera] ERRO: não foi possível abrir a câmera de índice {index}.")
        else:
            print(f"[Camera] Câmera aberta com sucesso (índice={index}, res={width}x{height}).")

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
                print("[Camera] Aviso: arquivo de calibração não contém 'camera_matrix' válida.")
            
            # Carrega coeficientes de distorção
            dist = calib_data.get("dist_coeffs", None)
            if dist:
                self.dist_coeffs = np.array(dist).astype(np.float32)
            else:
                print("[Camera] Aviso: arquivo de calibração não contém 'dist_coeffs'.")

            if self.camera_matrix is not None and self.dist_coeffs is not None:
                print("[Camera] Calibração carregada com sucesso!")
            else:
                print("[Camera] Calibração incompleta ou ausente, seguindo sem correção.")
        except FileNotFoundError:
            print(f"[Camera] Arquivo de calibração não encontrado: {calibration_file}")
        except Exception as e:
            print(f"[Camera] Erro ao carregar calibração: {e}")

    def get_frame(self):
        """
        Captura um frame da câmera. Aplica correção de distorção se os parâmetros de calibração
        estiverem disponíveis.
        
        :return: O frame corrigido (ou bruto, se não houver calibração) ou None se falhar.
        """
        if not self.cap or not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("[Camera] Falha ao capturar o frame!")
            return None

        # Se houver dados de calibração disponíveis, faz undistort
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            # Opção 1: Uso direto de cv2.undistort
            frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
        return frame

    def release(self):
        """
        Libera a câmera ao final do uso.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Câmera liberada.")
