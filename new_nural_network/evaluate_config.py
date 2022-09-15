class EvaluateConfig:
    def __init__(self, model_agent0,
                 model_agent1,
                 hidden_units_agent0,
                 hidden_units_agent1,
                 episodes):
        self.model_agent0 = model_agent0
        self.model_agent1 = model_agent1
        self.hidden_units_agent0 = hidden_units_agent0
        self.hidden_units_agent1 = hidden_units_agent1
        self.episodes = episodes