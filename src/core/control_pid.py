class MotionController:
    def __init__(self):
        # Parâmetros de PID, ZMP, etc.
        self.linear_pid = PIDController(kp=1.0, ki=0.0, kd=0.0)
        self.angular_pid = PIDController(kp=0.5, ki=0.0, kd=0.0)

    def step(self, v_linear, v_angular, dt, sensors_data):
        """
        Gera comandos (ex.: ângulo de juntas) para andar na direção desejada.
        Exemplo simplificado (ZMP real é mais complexo).
        """
        # Podemos combinar feedforward (zmp) + correção PD do torso + leitura do foot sensor
        linear_output = self.linear_pid.compute(v_linear, 0.0, dt)  # ex.: erro entre setpoint e "vel. atual"
        angular_output = self.angular_pid.compute(v_angular, 0.0, dt)
        # Gera sequência de juntas do humanoide...
        return (linear_output, angular_output)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.prev_error = 0

    def compute(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
