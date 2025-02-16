import math
import time

class OdometryState:
    def __init__(self):
        self.vel_x = 0.0  # Velocidade acumulada em X (m/s)
        self.vel_y = 0.0  # Velocidade acumulada em Y (m/s)
        self.last_time = None  # Armazena o último timestamp para calcular dt

    def compute_odometry_from_imu(self, perception):
        """
        Integra acelerações medidas pelo IMU para estimar deslocamento linear
        e usa a rotação (yaw) também do IMU para atualizar (dx, dy) em coordenadas globais.
        
        :param perception: Objeto Perception que tem:
        - yaw (float) atualizado
        - pitch, roll (caso queira usar)
        - imu_data_queue[-1] com accel=(ax, ay, az) em m/s^2 (idealmente)
        :param odom_state: Instância de OdometryState, contendo vel_x, vel_y e last_time
        :return: (dx, dy, dtheta)
        """
        # 1) Se não há leituras recentes, retorne zero
        if not perception.imu_data_queue:
            return (0.0, 0.0, 0.0)
        
        # Pega a última leitura do IMU
        data = perception.imu_data_queue[-1]
        now = data['timestamp']
        ax, ay, az = data['accel']  # Em m/s^2 (idealmente)
        gx, gy, gz = data['gyro']   # Em rad/s
        # yaw (do perception) já é filtrado (fuse_orientation)
        yaw = perception.yaw
        
        # 2) Cálculo de dt
        if self.last_time is None:
            self.last_time = now
            return (0.0, 0.0, 0.0)
        dt = now - self.last_time
        self.last_time = now
        
        # 3) Remover gravidade (parcialmente) se a Z do IMU estiver apontando sempre para cima.
        #    Exemplo simples: se az ~ 9.81, subtraímos 9.81 do az.
        #    Se o robô inclina, isso se complica. Aqui faremos algo simples:
        #    Se pitch e roll forem pequenos, assumimos ~9.81 em az.
        #    (Ajuste conforme necessidade)
        # Se o robô estiver inclinado, esse "hack" não é tão preciso.
        # ax, ay, az = filtrar_gravidade(ax, ay, az, perception.pitch, perception.roll)
        # Exemplo "grosseiro":
        az = az - 9.81  # remove gravidade, assumindo que Z aponta pra cima
        
        # 4) Transformar aceleração local (ax, ay) considerando o yaw para "frente do robô"
        # Se ax, ay forem no frame do robô (IMU), precisamos decidir:
        #   - Se yaw=0 => x local = x global, y local = y global
        #   - Caso contrário, rotacionar (ax, ay) pelo yaw.
        
        # Entretanto, muitos IMUs ficam "de lado" ou "inclinados" no robô, precisaria calibrar.
        # Exemplo simples:
        # ax_global = ax*cos(yaw) - ay*sin(yaw)
        # ay_global = ax*sin(yaw) + ay*cos(yaw)
        # Supondo que pitch e roll são pequenos.
        
        ax_global = ax * math.cos(yaw) - ay * math.sin(yaw)
        ay_global = ax * math.sin(yaw) + ay * math.cos(yaw)
        
        # 5) Integra a aceleração para atualizar velocidade
        self.vel_x += ax_global * dt
        self.vel_y += ay_global * dt
        
        # 6) Calcula deslocamento no intervalo dt
        dx = self.vel_x * dt
        dy = self.vel_y * dt
        
        # 7) Calcular variação de ângulo dtheta. 
        #    No caso humanoide, podemos usar gz (giroscópio em Z) como rotação principal
        dtheta = gz * dt  # Se gz corresponde a rotação em torno do eixo Z do robô
        
        # 8) Retorna (dx, dy, dtheta)
        return (dx, dy, dtheta)
