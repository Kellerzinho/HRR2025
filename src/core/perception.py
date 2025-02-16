import time
import numpy as np
from collections import deque

class Perception:
    def __init__(self, config):
        self.offsets = config.get('offsets', {})
        # Fila para sincronizar timestamps de IMU e imagem
        self.imu_data_queue = deque(maxlen=100)

    def update_imu_data(self, imu_data):
        """
        Recebe leituras de IMU + magnetômetro. Exemplo:
        imu_data = {
          'timestamp': time.time(),
          'accel': (ax, ay, az),
          'gyro': (gx, gy, gz),
          'mag': (mx, my, mz)
        }
        """
        self.imu_data_queue.append(imu_data)

    def fuse(self, frame_timestamp):
        """
        Retorna a estimativa de orientação no momento do frame,
        procurando dados de IMU com timestamp mais próximo.
        """
        # Exemplo muito simplificado
        best_match = None
        best_diff = float('inf')
        for data in self.imu_data_queue:
            diff = abs(data['timestamp'] - frame_timestamp)
            if diff < best_diff:
                best_diff = diff
                best_match = data

        if best_match:
            # Processar best_match com um EKF, por exemplo
            # ...
            yaw, pitch, roll = 0.0, 0.0, 0.0  # Exemplo
            # Use magnetômetro, giroscópio, etc. para compor
            return (yaw, pitch, roll)
        return (0,0,0)

    def compute_world_coords(self, detection, orientation):
        """
        Converte coordenadas de detecção na imagem para posição no mundo (x, y, z).
        Usa a orientação do robô (do EKF) + offsets de câmera.
        """
        # Com base em triângulo inverso, projeção, etc.
        # ...
        return (0.0, 0.0, 0.0)
