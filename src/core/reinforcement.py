class ReinforcementLearner:
    def __init__(self, state_dim, action_dim):
        # Inicializar rede neural, etc.
        pass

    def select_action(self, state):
        # Devolve a ação de maior Q-Value, ou amostra e-greedy.
        pass

    def train(self, transition):
        # transition = (state, action, reward, next_state, done)
        pass
