import math

class Navigator:
    def __init__(self):
        pass

    def plan_path(self, current_pose, target_pose, obstacles=None):
        """
        Retorna uma lista de waypoints, ou uma direção para avançar, ignorando colisões simples.
        Exemplo minimalista
        """
        return [target_pose]

    def compute_velocity(self, current_pose, next_waypoint):
        """
        Retorna (v_linear, v_angular) para PID ou controle ZMP
        """
        # Exemplo: v_angular para alinhar ângulo, v_linear se ângulo < threshold
        dx = next_waypoint[0] - current_pose[0]
        dy = next_waypoint[1] - current_pose[1]
        angle_to_goal = math.atan2(dy, dx)
        angle_diff = angle_to_goal - current_pose[2]  # supor 2D
        # Simples:
        v_angular = angle_diff
        dist = math.sqrt(dx*dx + dy*dy)
        v_linear = min(dist, 0.3) # limita velocidade
        return (v_linear, v_angular)
