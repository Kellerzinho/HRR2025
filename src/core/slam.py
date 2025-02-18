import math
import random
import numpy as np

class MonteCarloLocalization:
    """
    Implementa um Filtro de Partículas (Monte Carlo Localization)
    para um campo de futebol de robôs.
    """
    def __init__(self, field_map, num_particles=200, x_var=0.1, y_var=0.1, theta_var=0.1):
        self.field_map = field_map
        self.width = field_map.get('width', 9.0)
        self.height = field_map.get('height', 6.0)

        # Goleiras (coordenadas)
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

        # Inicializa partículas
        self.particles = self._initialize_particles()

    def _initialize_particles(self):
        """
        Exemplo: se o robô inicia no (0,0).
        """
        particles = []
        x0, y0 = 0.0, 0.0
        for _ in range(self.num_particles):
            x = random.gauss(x0, 0.2)
            y = random.gauss(y0, 0.2)
            theta = random.uniform(-math.pi, math.pi)
            w = 1.0 / self.num_particles
            particles.append([x, y, theta, w])
        return particles

    def update(self, odom, dt, world_positions):
        self._predict(odom, dt)
        self._correct(world_positions)
        self._resample()

    def _predict(self, odom, dt):
        """
        odom = (dx, dy, dtheta) no referencial do robô
        """
        dx, dy, dtheta = odom
        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            global_dx = dx * cos_t - dy * sin_t
            global_dy = dx * sin_t + dy * cos_t

            x_new = x + global_dx
            y_new = y + global_dy
            theta_new = self._normalize_angle(theta + dtheta)

            # Injeta ruído
            x_new += random.gauss(0, self.x_var * dt)
            y_new += random.gauss(0, self.y_var * dt)
            theta_new += random.gauss(0, self.theta_var * dt)

            # Restringe ao campo
            x_new = max(0, min(self.width, x_new))
            y_new = max(0, min(self.height, y_new))
            theta_new = self._normalize_angle(theta_new)

            self.particles[i] = [x_new, y_new, theta_new, w]

    def _correct(self, world_positions):
        """
        world_positions pode ter:
          - "goal_measure": [(dist, angle, side), (dist, angle, side), ...]
          - "penaltycross_measure": [(dist, angle, side), ...]
          - "center_circle_measure": (dist, angle) (opcional)
        """
        # Desvios-padrão
        dist_std_goal = 0.5
        angle_std_goal = 0.2
        dist_std_penalty = 0.3
        angle_std_penalty = 0.1
        dist_std_center = 0.3
        angle_std_center = 0.1

        total_weight = 0.0

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]
            new_weight = w

            # 1) Se houver medições de gol
            #    goal_measure: lista de (dist, angle, side)
            if "goal_measure" in world_positions:
                for (dist_med, angle_med, side) in world_positions["goal_measure"]:
                    # Decide qual gol usar
                    if side == "left":
                        gx, gy = self.left_goal_x, self.left_goal_y
                        dist_std = dist_std_goal
                        angle_std = angle_std_goal
                    else:
                        gx, gy = self.right_goal_x, self.right_goal_y
                        dist_std = dist_std_goal
                        angle_std = angle_std_goal

                    # Dist e angle esperados se a partícula está em (x, y, theta)
                    dx = gx - x
                    dy = gy - y
                    expected_dist = math.sqrt(dx*dx + dy*dy)
                    expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                    dist_error = dist_med - expected_dist
                    angle_error = self._normalize_angle(angle_med - expected_angle)

                    w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std**2)))
                    w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std**2)))
                    new_weight *= (w_dist * w_angle)

            # 2) Se houver medições de penalty cross
            #    penaltycross_measure: lista de (dist, angle, side)
            if "penaltycross_measure" in world_positions:
                for (dist_med, angle_med, side) in world_positions["penaltycross_measure"]:
                    if side == "left":
                        px, py = self.left_penalty_x, self.left_penalty_y
                    else:
                        px, py = self.right_penalty_x, self.right_penalty_y

                    dx = px - x
                    dy = py - y
                    expected_dist = math.sqrt(dx*dx + dy*dy)
                    expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                    dist_error = dist_med - expected_dist
                    angle_error = self._normalize_angle(angle_med - expected_angle)

                    w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_penalty**2)))
                    w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_penalty**2)))
                    new_weight *= (w_dist * w_angle)

            # 3) Se houver medições do círculo central (opcional)
            #    "center_circle_measure": pode ser (dist, angle)
            if "center_circle_measure" in world_positions:
                dist_cc, angle_cc = world_positions["center_circle_measure"]
                dx = self.center_x - x
                dy = self.center_y - y
                expected_dist = math.sqrt(dx*dx + dy*dy)
                expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)

                dist_error = dist_cc - expected_dist
                angle_error = self._normalize_angle(angle_cc - expected_angle)

                w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_center**2)))
                w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_center**2)))
                new_weight *= (w_dist * w_angle)

            # Atualiza o peso
            self.particles[i][3] = new_weight
            total_weight += new_weight

        # Normaliza
        if total_weight > 1e-6:
            for i in range(len(self.particles)):
                self.particles[i][3] /= total_weight
        else:
            self.particles = self._initialize_particles()

    def _resample(self):
        new_particles = []
        weights = [p[3] for p in self.particles]
        cumsum_weights = np.cumsum(weights)
        if cumsum_weights[-1] < 1e-9:
            self.particles = self._initialize_particles()
            return

        for _ in range(self.num_particles):
            r = random.random() * cumsum_weights[-1]
            idx = np.searchsorted(cumsum_weights, r)
            x, y, theta, w = self.particles[idx]
            new_particles.append([x, y, theta, 1.0 / self.num_particles])

        self.particles = new_particles

    def get_best_estimate(self):
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
        while angle > math.pi:
            angle -= 2*math.pi
        while angle <= -math.pi:
            angle += 2*math.pi
        return angle
