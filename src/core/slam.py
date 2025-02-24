import cv2
import math
import random
import numpy as np

class MonteCarloLocalization:
    """
    Implementa um Filtro de Partículas (Monte Carlo Localization)
    para um campo de futebol de robôs, agora incorporando um componente
    de odometria visual usando ORB e correspondência de features.
    """
    def __init__(self, field_map, num_particles=200, x_var=0.1, y_var=0.1, theta_var=0.1):
        self.field_map = field_map
        self.width = field_map.get('width', 9.0)
        self.height = field_map.get('height', 6.0)

        # Coordenadas dos landmarks fixos (goleiras, círculo central, penalty crosses)
        self.left_goal_x = field_map.get('left_goal_x', 0.0)
        self.left_goal_y = field_map.get('left_goal_y', 3.0)
        self.right_goal_x = field_map.get('right_goal_x', 9.0)
        self.right_goal_y = field_map.get('right_goal_y', 3.0)
        self.center_x = field_map.get('center_circle_x', 4.5)
        self.center_y = field_map.get('center_circle_y', 3.0)
        self.left_penalty_x = field_map.get('left_penaltycross_x', 1.5)
        self.left_penalty_y = field_map.get('left_penaltycross_y', 3.0)
        self.right_penalty_x = field_map.get('right_penaltycross_x', 7.5)
        self.right_penalty_y = field_map.get('right_penaltycross_y', 3.0)

        self.num_particles = num_particles
        self.x_var = x_var
        self.y_var = y_var
        self.theta_var = theta_var

        # Inicializa as partículas (por exemplo, começando em (0,0))
        self.particles = self._initialize_particles()

        # Armazena o frame anterior para cálculo de visual odometry
        self.prev_frame = None

    def _initialize_particles(self):
        particles = []
        # Exemplo: inicializa em (0,0); se o robô realmente começar em outro lugar, ajuste aqui.
        x0, y0 = 0.0, 0.0
        for _ in range(self.num_particles):
            x = random.gauss(x0, 0.2)
            y = random.gauss(y0, 0.2)
            theta = random.uniform(-math.pi, math.pi)
            w = 1.0 / self.num_particles
            particles.append([x, y, theta, w])
        return particles

    def compute_visual_odometry(self, prev_frame, curr_frame):
        """
        Calcula a odometria visual entre dois frames usando ORB.
        Retorna uma tupla (dx, dy, dtheta) representando a transformação estimada.
        """
        # Cria detector ORB
        orb = cv2.ORB_create(nfeatures=1000)
        kp1, des1 = orb.detectAndCompute(prev_frame, None)
        kp2, des2 = orb.detectAndCompute(curr_frame, None)

        if des1 is None or des2 is None:
            print("[SLAM] ERRO: Não foi possível extrair features ORB de um dos frames.")
            return None

        # Emparelhador brute-force com norma Hamming (para ORB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        if not matches:
            print("[SLAM] ERRO: Não foi possível encontrar matches ORB entre os frames.")
            return None

        # Ordena os matches pela distância
        matches = sorted(matches, key=lambda m: m.distance)
        # Seleciona os melhores matches
        num_good = int(len(matches) * 0.5)
        good_matches = matches[:num_good]

        # Extrai as coordenadas dos keypoints correspondentes
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calcula a matriz essencial usando os pontos e a camera_matrix (supondo que ela esteja definida)
        if not hasattr(self, 'camera_matrix') or self.camera_matrix is None:
            print("[SLAM] ERRO: camera_matrix não definida para odometria visual.")
            return None

        E, mask = cv2.findEssentialMat(pts2, pts1, self.camera_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print("[SLAM] ERRO: Não foi possível calcular a matriz essencial para odometria visual.")
            return None

        # Recupera a pose (rotação e translação)
        _, R, t, mask_pose = cv2.recoverPose(E, pts2, pts1, self.camera_matrix)

        # Aproximação simples para dtheta: calculamos o ângulo da rotação no plano (assumindo pequenas rotações)
        dtheta = math.atan2(R[1, 0], R[0, 0])
        # t é um vetor 3x1; assumimos que a movimentação é no plano
        dx = t[0][0]
        dy = t[1][0]
        return (dx, dy, dtheta)

    def _predict(self, odom, dt, visual_odom=None):
        """
        Atualiza cada partícula usando odometria do IMU (odom) e, se disponível, 
        combina com a odometria visual (visual_odom).
        odom: (dx, dy, dtheta) do IMU
        visual_odom: (dx_v, dy_v, dtheta_v) do método de odometria visual
        """
        # Combina odometria: podemos fazer uma média ponderada se ambas estiverem disponíveis
        if visual_odom is not None:
            # Aqui, atribuímos pesos (exemplo: 0.5 para cada fonte)
            dx = 0.5 * odom[0] + 0.5 * visual_odom[0]
            dy = 0.5 * odom[1] + 0.5 * visual_odom[1]
            dtheta = 0.5 * odom[2] + 0.5 * visual_odom[2]
        else:
            dx, dy, dtheta = odom

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]
            # Converte (dx, dy) do referencial local para global com base no theta da partícula
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            global_dx = dx * cos_t - dy * sin_t
            global_dy = dx * sin_t + dy * cos_t

            x_new = x + global_dx
            y_new = y + global_dy
            theta_new = self._normalize_angle(theta + dtheta)

            # Injeta ruído gaussiano
            x_new += random.gauss(0, self.x_var * dt)
            y_new += random.gauss(0, self.y_var * dt)
            theta_new += random.gauss(0, self.theta_var * dt)

            # Garante que a partícula permaneça dentro dos limites do campo
            x_new = max(0, min(self.width, x_new))
            y_new = max(0, min(self.height, y_new))
            theta_new = self._normalize_angle(theta_new)

            self.particles[i] = [x_new, y_new, theta_new, w]

    def _correct(self, world_positions):
        # (Método de correção permanece igual ao que você já implementou)
        # Aqui, cada medições (como 'goal_measure', 'penaltycross_measure', etc.)
        # são usadas para ajustar os pesos das partículas.
        # [Mantemos o código existente de _correct]
        dist_std_goal = 0.5
        angle_std_goal = 0.2
        dist_std_center = 0.3
        angle_std_center = 0.1
        dist_std_penalty = 0.3
        angle_std_penalty = 0.1
        total_weight = 0.0

        if not world_positions:
            return

        for i in range(len(self.particles)):
            x, y, theta, w = self.particles[i]
            new_weight = w

            if "goal_measure" in world_positions:
                for (dist_med, angle_med, side) in world_positions["goal_measure"]:
                    if side == "left":
                        gx, gy = self.left_goal_x, self.left_goal_y
                    else:
                        gx, gy = self.right_goal_x, self.right_goal_y
                    dx = gx - x
                    dy = gy - y
                    expected_dist = math.sqrt(dx*dx + dy*dy)
                    expected_angle = self._normalize_angle(math.atan2(dy, dx) - theta)
                    dist_error = dist_med - expected_dist
                    angle_error = self._normalize_angle(angle_med - expected_angle)
                    w_dist = math.exp(-0.5 * (dist_error**2 / (dist_std_goal**2)))
                    w_angle = math.exp(-0.5 * (angle_error**2 / (angle_std_goal**2)))
                    new_weight *= (w_dist * w_angle)

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

            self.particles[i][3] = new_weight
            total_weight += new_weight

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
            angle -= 2 * math.pi
        while angle <= -math.pi:
            angle += 2 * math.pi
        return angle

    def update(self, odom, dt, world_positions, curr_frame=None):
        """
        Atualiza o filtro de partículas integrando:
         - Odometria do IMU (odom)
         - Odometry visual se curr_frame estiver disponível e self.prev_frame estiver definido.
         - Correção via medições dos landmarks (world_positions)
        """
        visual_odom = None
        if curr_frame is not None and self.prev_frame is not None:
            visual_odom = self.compute_visual_odometry(self.prev_frame, curr_frame)
        # Atualiza a predição usando a combinação (aqui, média simples)
        self._predict(odom, dt, visual_odom)
        self._correct(world_positions)
        self._resample()

        # Atualiza o frame anterior para o próximo cálculo de odometria visual
        if curr_frame is not None:
            self.prev_frame = curr_frame.copy()
