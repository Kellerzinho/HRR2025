# src/core/slam.py

import math
import random
import numpy as np

class MonteCarloLocalization:
    """
    Implementa um Filtro de Partículas (Monte Carlo Localization)
    para um campo de futebol de robôs.
    """
    def __init__(self, field_map, num_particles=200, x_var=0.1, y_var=0.1, theta_var=0.1):
        """
        :param field_map: dict com informações do campo, por ex.:
            {
              "width": 9.0,
              "height": 6.0,
              "left_goal_x": 0.0,
              "left_goal_y": 3.0,
              "right_goal_x": 9.0,
              "right_goal_y": 3.0,
              "center_circle_x": 4.5,
              "center_circle_y": 3.0,
              "left_penaltycross_x": 1.5,
              "left_penaltycross_y": 3.0,
              "right_penaltycross_x": 7.5,
              "right_penaltycross_y": 3.0
            }
        :param num_particles: número de partículas no filtro
        :param x_var, y_var, theta_var: variâncias (ou desvios-padrão) para injetar ruído
                                        na etapa de predição e reamostragem
        """
        self.field_map = field_map
        self.width = field_map.get('width', 9.0)
        self.height = field_map.get('height', 6.0)

        # Goleiras
        self.left_goal_x = field_map.get('left_goal_x', 0.0)
        self.left_goal_y = field_map.get('left_goal_y', 3.0)
        self.right_goal_x = field_map.get('right_goal_x', 9.0)
        self.right_goal_y = field_map.get('right_goal_y', 3.0)

        # Círculo central
        self.center_x = field_map.get('center_circle_x', 4.5)
        self.center_y = field_map.get('center_circle_y', 3.0)

        # Penalty crosses
        self.left_penalty_x = field_map.get('left_penaltycross_x', 1.5)
        self.left_penalty_y = field_map.get('left_penaltycross_y', 3.0)
        self.right_penalty_x = field_map.get('right_penaltycross_x', 7.5)
        self.right_penalty_y = field_map.get('right_penaltycross_y', 3.0)

        self.num_particles = num_particles
        self.x_var = x_var
        self.y_var = y_var
        self.theta_var = theta_var

        # Lista de partículas: cada partícula = [x, y, theta, weight]
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        """
        Inicializa partículas com distribuição normal em torno de uma posição inicial conhecida.
        Por exemplo, se o robô iniciar no meio do campo (4.5, 3.0):
        """
        particles = []
        x0, y0 = 4.5, 3.0  # Posição inicial desejada
        for _ in range(self.num_particles):
            # Usa uma distribuição normal com desvio padrão pequeno
            x = random.gauss(x0, 0.2)
            y = random.gauss(y0, 0.2)
            theta = random.uniform(-math.pi, math.pi)
            w = 1.0 / self.num_particles
            particles.append([x, y, theta, w])
        return particles

    def update(self, odom, dt, world_positions):
        """
        Atualiza o filtro de partículas (predição e correção) e reamostra.

        :param odom: (dx, dy, dtheta) - deslocamento estimado do robô nesse intervalo
                     pode vir de IMU + contagem de passos, etc.
        :param dt: passo de tempo
        :param world_positions: dict retornado pela percepção contendo medições, por ex.:
           {
             'left_goal_measure': (dist, angle),
             'right_goal_measure': (dist, angle),
             'center_circle_measure': (dist, angle),
             'left_penaltycross_measure': (dist, angle),
             'right_penaltycross_measure': (dist, angle)
             ...
           }
        """
        self._predict(odom, dt)
        self._correct(world_positions)
        self._resample()

    def _predict(self, odom, dt):
        """
        Etapa de predição: aplica odometria em cada partícula + ruído.
        odom = (dx, dy, dtheta) no referencial do robô.
        """
        dx, dy, dtheta = odom

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]

            # Converte (dx, dy) do referencial local para global baseado em theta
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            global_dx = dx * cos_t - dy * sin_t
            global_dy = dx * sin_t + dy * cos_t

            x_new = x + global_dx
            y_new = y + global_dy
            theta_new = theta + dtheta

            # Normaliza ângulo
            theta_new = self._normalize_angle(theta_new)

            # Injeta ruído gaussiano
            x_new += random.gauss(0, self.x_var * dt)
            y_new += random.gauss(0, self.y_var * dt)
            theta_new += random.gauss(0, self.theta_var * dt)

            # Garante que a partícula fique no campo
            x_new = max(0, min(self.width, x_new))
            y_new = max(0, min(self.height, y_new))
            theta_new = self._normalize_angle(theta_new)

            self.particles[i] = [x_new, y_new, theta_new, w]

    def _correct(self, world_positions):
        """
        Ajusta o peso (weight) de cada partícula com base em múltiplas medições:
          - left_goal_measure
          - right_goal_measure
          - center_circle_measure
          - left_penaltycross_measure
          - right_penaltycross_measure
        Cada medição é (dist, angle).
        """
        # Desvios-padrão (exemplo)
        dist_std_goal = 0.5
        angle_std_goal = 0.2

        dist_std_center = 0.3
        angle_std_center = 0.1

        dist_std_penalty = 0.3
        angle_std_penalty = 0.1

        total_weight = 0.0

        # Se não há medições, não faz correção
        if not world_positions:
            return

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]
            new_weight = w

            # 1) Left Goal
            left_goal_measure = world_positions.get('left_goal_measure')
            if left_goal_measure:
                dist_med, angle_med = left_goal_measure
                dx = self.left_goal_x - x
                dy = self.left_goal_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_goal**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_goal**2)))
                new_weight *= (w_dist * w_angle)

            # 2) Right Goal
            right_goal_measure = world_positions.get('right_goal_measure')
            if right_goal_measure:
                dist_med, angle_med = right_goal_measure
                dx = self.right_goal_x - x
                dy = self.right_goal_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_goal**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_goal**2)))
                new_weight *= (w_dist * w_angle)

            # 3) Center Circle
            center_measure = world_positions.get('center_circle_measure')
            if center_measure:
                dist_med, angle_med = center_measure
                dx = self.center_x - x
                dy = self.center_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_center**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_center**2)))
                new_weight *= (w_dist * w_angle)

            # 4) Left Penalty Cross
            left_pen_measure = world_positions.get('left_penaltycross_measure')
            if left_pen_measure:
                dist_med, angle_med = left_pen_measure
                dx = self.left_penalty_x - x
                dy = self.left_penalty_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_penalty**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_penalty**2)))
                new_weight *= (w_dist * w_angle)

            # 5) Right Penalty Cross
            right_pen_measure = world_positions.get('right_penaltycross_measure')
            if right_pen_measure:
                dist_med, angle_med = right_pen_measure
                dx = self.right_penalty_x - x
                dy = self.right_penalty_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_penalty**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_penalty**2)))
                new_weight *= (w_dist * w_angle)

            # Atualiza peso da partícula
            self.particles[i][3] = new_weight
            total_weight += new_weight

        # Normaliza pesos
        if total_weight > 1e-6:
            for i in range(len(self.particles)):
                self.particles[i][3] /= total_weight
        else:
            # Se pesos degeneraram, reinicializa
            self.particles = self._initialize_particles()

    def _resample(self):
        """
        Reamostragem (Roulette Wheel).
        """
        new_particles = []
        weights = [p[3] for p in self.particles]
        cumsum_weights = np.cumsum(weights)
        if cumsum_weights[-1] < 1e-9:
            # Se soma de pesos ~ zero, reinicializa
            self.particles = self._initialize_particles()
            return

        for _ in range(self.num_particles):
            r = random.random() * cumsum_weights[-1]
            idx = np.searchsorted(cumsum_weights, r)
            x, y, theta, w = self.particles[idx]
            new_particles.append([x, y, theta, 1.0 / self.num_particles])

        self.particles = new_particles

    def get_best_estimate(self):
        """
        Retorna a média ponderada (x, y, theta).
        """
        x_est = 0.0
        y_est = 0.0
        sin_th = 0.0
        cos_th = 0.0
        total_w = 0.0

        for (x, y, theta, w) in self.particles:
            x_est += x * w
            y_est += y * w
            sin_th += math.sin(theta) * w
            cos_th += math.cos(theta) * w
            total_w += w

        if total_w > 1e-6:
            x_est /= total_w
            y_est /= total_w
            sin_th /= total_w
            cos_th /= total_w
            theta_est = math.atan2(sin_th, cos_th)
            return (x_est, y_est, theta_est)
        else:
            return (0.0, 0.0, 0.0)

    def _normalize_angle(self, angle):
        """
        Ajusta ângulo para -pi..+pi
        """
        while angle > math.pi:
            angle -= 2*math.pi
        while angle <= -math.pi:
            angle += 2*math.pi
        return angle
