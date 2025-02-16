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
              "goal_x": 9.0,
              "goal_y": 3.0
            }
        :param num_particles: número de partículas no filtro
        :param x_var, y_var, theta_var: variâncias (ou desvios-padrão) para injetar ruído
                                        na etapa de predição e reamostragem
        """
        self.field_map = field_map
        self.width = field_map.get('width', 9.0)
        self.height = field_map.get('height', 6.0)

        self.goal_x = field_map.get('goal_x', 9.0)
        self.goal_y = field_map.get('goal_y', 3.0)
        self.centercircle_x = field_map.get('centercircle_x', 4.5)
        self.centercircle_y = field_map.get('centercircle_y', 3.0)
        self.penaltycross_x = field_map.get('penaltycross_x', 7.5)
        self.penaltycross_y = field_map.get('penaltycross_y', 3.0)

        self.num_particles = num_particles
        self.x_var = x_var
        self.y_var = y_var
        self.theta_var = theta_var

        # Lista de partículas: cada partícula = (x, y, theta, weight)
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        """
        Inicializa partículas de forma uniforme pelo campo e ângulo aleatório.
        Peso inicial = 1 / num_particles.
        """
        particles = []
        for _ in range(self.num_particles):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
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
        :param world_positions: dict retornado pela percepção. Por ex.:
           {
             'goal_measure': (distance_measured, angle_measured),
             ...
           }
           Aqui vamos focar no 'goal_measure'.
        """
        # 1) Predição
        self._predict(odom, dt)

        # 2) Correção - compara medições do "goal" com o valor esperado
        self._correct(world_positions)

        # 3) Reamostragem
        self._resample()

    def _predict(self, odom, dt):
        """
        Etapa de predição: aplica odometria em cada partícula + ruído.
        odom = (dx, dy, dtheta) no referencial do robô.
        """
        dx, dy, dtheta = odom

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]

            # Aplica deslocamento no referencial local da partícula
            # 1) Gira o vetor (dx, dy) pelo theta da partícula:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            global_dx = dx * cos_t - dy * sin_t
            global_dy = dx * sin_t + dy * cos_t

            # 2) Atualiza posição
            x_new = x + global_dx
            y_new = y + global_dy
            theta_new = theta + dtheta

            # Normaliza ângulo
            theta_new = self._normalize_angle(theta_new)

            # Injeta ruído gaussiano
            x_new += random.gauss(0, self.x_var * dt)
            y_new += random.gauss(0, self.y_var * dt)
            theta_new += random.gauss(0, self.theta_var * dt)

            # Garante que a partícula permaneça dentro do campo (opcional)
            x_new = max(0, min(self.width, x_new))
            y_new = max(0, min(self.height, y_new))
            theta_new = self._normalize_angle(theta_new)

            self.particles[i] = [x_new, y_new, theta_new, w]

    def _correct(self, world_positions):
        """
        Ajusta o peso (weight) de cada partícula com base em múltiplas medições de landmarks,
        por exemplo:
          - 'goal_measure': (dist_med, angle_med)
          - 'centercircle_measure': (dist_med, angle_med)
          - 'penaltycross_measure': (dist_med, angle_med)
        Cada medição é comparada com o valor esperado (dist, angle) se o robô estivesse na pose da partícula.
        """
        # Defina incertezas (desvios-padrão) ou use valores diferentes para cada landmark
        dist_std_goal = 0.5
        angle_std_goal = 0.2

        dist_std_center = 0.3
        angle_std_center = 0.1

        dist_std_penalty = 0.3
        angle_std_penalty = 0.1

        total_weight = 0.0

        # Se não há medições, não faz correção
        if not any(key.endswith('_measure') for key in world_positions.keys()):
            return

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]

            # Começamos com o peso atual da partícula
            new_weight = w

            # --------------------------------------------------
            # 1) GOAL MEASURE
            # --------------------------------------------------
            goal_measure = world_positions.get('goal_measure', None)
            if goal_measure:
                dist_med, angle_med = goal_measure

                # Cálculo do esperado:
                dx = self.goal_x - x
                dy = self.goal_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                # Likelihood
                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_goal**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_goal**2)))

                new_weight *= (w_dist * w_angle)

            # --------------------------------------------------
            # 2) CENTER CIRCLE MEASURE
            # --------------------------------------------------
            center_measure = world_positions.get('centercircle_measure', None)
            if center_measure:
                dist_med, angle_med = center_measure

                dx = self.centercircle_x - x
                dy = self.centercircle_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                # Likelihood
                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_center**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_center**2)))

                new_weight *= (w_dist * w_angle)

            # --------------------------------------------------
            # 3) PENALTYCROSS MEASURE
            # --------------------------------------------------
            penalty_measure = world_positions.get('penaltycross_measure', None)
            if penalty_measure:
                dist_med, angle_med = penalty_measure

                dx = self.penaltycross_x - x
                dy = self.penaltycross_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_med - expected_dist
                angle_error = self._normalize_angle(angle_med - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_penalty**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_penalty**2)))

                new_weight *= (w_dist * w_angle)

            # Atualiza o peso da partícula
            self.particles[i][3] = new_weight
            total_weight += new_weight

        # Normaliza pesos
        if total_weight > 1e-6:
            for i in range(len(self.particles)):
                self.particles[i][3] /= total_weight
        else:
            # Se pesos degeneraram, reinicializa
            for i in range(len(self.particles)):
                x, y, th, w = self.particles[i]
                self.particles[i][3] = 1.0 / self.num_particles

    def _resample(self):
        """
        Reamostragem (Stratified ou Systematic). Aqui usamos uma abordagem simples de 'roulette wheel'.
        """
        new_particles = []
        weights = [p[3] for p in self.particles]
        cumsum_weights = np.cumsum(weights)
        if cumsum_weights[-1] < 1e-9:
            # Se a soma de pesos é quase zero, reinicializa
            self.particles = self._initialize_particles()
            return

        for _ in range(self.num_particles):
            r = random.random() * cumsum_weights[-1]
            idx = np.searchsorted(cumsum_weights, r)
            # Copia partícula idx e faz pequena variação
            x, y, theta, w = self.particles[idx]
            new_particles.append([x, y, theta, 1.0 / self.num_particles])

        self.particles = new_particles

    def get_best_estimate(self):
        """
        Retorna a média ponderada (x, y, theta) ou a partícula de maior peso.
        """
        # Opção 1: média ponderada
        # soma(x_i * w_i), soma(y_i * w_i), ...
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
            # Em caso de degeneração
            return (0.0, 0.0, 0.0)

    def _normalize_angle(self, angle):
        """
        Ajusta o ângulo para estar entre -pi e +pi
        """
        while angle > math.pi:
            angle -= 2*math.pi
        while angle <= -math.pi:
            angle += 2*math.pi
        return angle
