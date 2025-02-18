# src/core/ball_tracker.py

import numpy as np

class BallTracker:
    """
    Rastreia a bola usando um Filtro de Kalman simples em 2D.
    Estado = [x, y, vx, vy]^T
    """

    def __init__(self, dt=0.02, process_std=0.5, measurement_std=0.2):
        """
        :param dt: intervalo de tempo (em s) entre atualizações (exemplo: 0.02 se ~50Hz).
        :param process_std: desvio-padrão do ruído de processo (quanto a bola muda de velocidade).
        :param measurement_std: desvio-padrão do ruído de medição (erro na detecção).
        """
        self.dt = dt
        
        # Estado inicial [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))  # posição e velocidade inicial = 0
        # Matriz de covariância inicial (incerteza grande)
        self.P = np.eye(4) * 1000.0

        # Matriz de transição de estados (movimento constante)
        # x_k+1 = F * x_k
        self.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1,     0],
            [0, 0, 0,     1]
        ])

        # Matriz de controle (se não controlamos a bola, pode ser zero)
        self.B = np.zeros((4, 4))

        # Matriz de observação (medimos x e y diretamente)
        # z = H * x, onde z = [x, y]^T
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Ruído de processo Q (assume aceleração pequena)
        self.Q = np.eye(4) * (process_std**2)

        # Ruído de medição R (erro na detecção)
        self.R = np.eye(2) * (measurement_std**2)

        self.last_update_time = None

    def predict(self, dt=None):
        """
        Predição de estado (movimento), se dt for fornecido, atualiza self.F.
        """
        if dt is not None:
            self.dt = dt
            self.F[0, 2] = self.dt
            self.F[1, 3] = self.dt

        # x = F*x
        self.x = self.F.dot(self.x)
        # P = F*P*F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, z):
        """
        Atualiza estado com a medição z = [x_meas, y_meas].
        """
        z = np.array(z).reshape((2, 1))  # Garantindo formato coluna
        # y = z - H*x
        y = z - self.H.dot(self.x)
        # S = H*P*H^T + R
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        # K = P*H^T * inv(S)
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        # x = x + K*y
        self.x = self.x + K.dot(y)
        # P = (I - K*H)*P
        I = np.eye(4)
        self.P = (I - K.dot(self.H)).dot(self.P)

    def get_state(self):
        """
        Retorna (x, y, vx, vy) estimados.
        """
        return self.x.flatten()

    def reset(self, x_init, y_init):
        """
        Redefine o estado inicial, se por acaso quisermos resetar.
        """
        self.x = np.array([x_init, y_init, 0, 0], dtype=np.float32).reshape((4, 1))
        self.P = np.eye(4) * 10.0
