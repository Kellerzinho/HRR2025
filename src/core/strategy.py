class Strategy:
    def __init__(self, role="atacante"):
        self.role = role
        self.state = "BUSCAR_BOLA"  # estado inicial

    def decide(self, world_state):
        """
        world_state pode conter:
        {
          'robot_pose': (x, y, theta),
          'ball_position': (x_b, y_b),
          'goal_position': (x_g, y_g),
          'obstacles': [...],
          ...
        }
        Retorna ação e alvo
        """
        if self.role == "atacante":
            return self.atacante_behavior(world_state)
        elif self.role == "goleiro":
            return self.goleiro_behavior(world_state)
        elif self.role == "zagueiro":
            return self.zagueiro_behavior(world_state)
        else:
            return ("idle", None)

    def atacante_behavior(self, ws):
        # Exemplo simples:
        if ws.get('ball_position'):
            ball_x, ball_y = ws['ball_position']
            # se estou longe da bola, vou até a bola
            return ("move_to", (ball_x, ball_y))
        else:
            # se não vejo bola, giro a cabeça ou campo
            return ("search", None)

    def goleiro_behavior(self, ws):
        # Futuro
        return ("idle", None)

    def zagueiro_behavior(self, ws):
        # Futuro
        return ("idle", None)
