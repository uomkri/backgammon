from new_nural_network.train_config import TrainConfig
from new_nural_network.utils import args_train


def start_train():
    args_for_train = TrainConfig()
    args_train(args_for_train)


if __name__ == "__main__":
    start_train()
