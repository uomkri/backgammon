class TrainConfig:
    def __init__(self, save_pass="saved/short",
                 steps_before_saving=10000,
                 episodes=1500000,
                 init_zero_weight=0,
                 lr=0.1,
                 hidden_units=160,
                 lamda=0.9,
                 model_location="saved/short/short__20220523_1201_23_805202_480000.tar",
                 experiment_name='short_',
                 model_type='nn',
                 seed=123,
                 game_type='short'):
        self.save_path = save_pass
        self.save_step = steps_before_saving
        self.episodes = episodes
        self.init_weights = init_zero_weight
        self.lr = lr
        self.hidden_units = hidden_units
        self.lamda = lamda
        self.model = model_location
        self.name = experiment_name
        self.type = model_type
        self.seed = seed
        self.game_type = game_type