import os
import gym
import sys
from agents import TDAgent, evaluate_agents
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from new_nural_network.model import TDGammon
# from torch.utils.tensorboard import SummaryWriter

#  tensorboard --logdir=runs/ --host localhost --port 8001


def write_file(path, **kwargs):
    with open('{}/parameters.txt'.format(path), 'w+') as file:
        print("Parameters:")
        for key, value in kwargs.items():
            file.write("{}={}\n".format(key, value))
            print("{}={}".format(key, value))
        print()


def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        print("The path {} doesn't exists".format(path))
        sys.exit()


# ==================================== TRAINING PARAMETERS ===================================

def args_train(args):
    save_step = args.save_step
    save_path = None
    n_episodes = args.episodes
    init_weights = args.init_weights
    lr = args.lr
    hidden_units = args.hidden_units
    lamda = args.lamda
    name = args.name
    model_type = args.type
    seed = args.seed
    game_type = args.game_type

    eligibility = False
    optimizer = None

    net = TDGammon(hidden_units=hidden_units, lr=lr, lamda=lamda, init_weights=init_weights, seed=seed, game_type=game_type)
    eligibility = True
    env = gym.make('gym_backgammon:backgammon-v1', game_type=game_type)

    if args.model and path_exists(args.model):
        # assert os.path.exists(args.model), print("The path {} doesn't exists".format(args.model))
        net.load(checkpoint_path=args.model, optimizer=optimizer, eligibility_traces=eligibility)

    if args.save_path and path_exists(args.save_path):
        # assert os.path.exists(args.save_path), print("The path {} doesn't exists".format(args.save_path))
        save_path = args.save_path

        write_file(
            save_path, save_path=args.save_path, command_line_args=args, type=model_type, hidden_units=hidden_units, init_weights=init_weights, alpha=net.lr, lamda=net.lamda,
            n_episodes=n_episodes, save_step=save_step, start_episode=net.start_episode, name_experiment=name, env=env.spec.id, restored_model=args.model, seed=seed,
            eligibility=eligibility, optimizer=optimizer, modules=[module for module in net.modules()]
        )

    net.train_agent(env=env, n_episodes=n_episodes, save_path=save_path, save_step=save_step, eligibility=eligibility, name_experiment=name)


# =================================== EVALUATE PARAMETERS ====================================

def args_evaluate(args):
    model_agent0 = args.model_agent0
    model_agent1 = args.model_agent1
    hidden_units_agent0 = args.hidden_units_agent0
    hidden_units_agent1 = args.hidden_units_agent1
    n_episodes = args.episodes

    if path_exists(model_agent0) and path_exists(model_agent1):
        assert os.path.exists(model_agent0), print("The path {} doesn't exists".format(model_agent0))
        assert os.path.exists(model_agent1), print("The path {} doesn't exists".format(model_agent1))

        net0 = TDGammon(hidden_units=hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
        net1 = TDGammon(hidden_units=hidden_units_agent1, lr=0.1, lamda=None, init_weights=False)
        env = gym.make('gym_backgammon:backgammon-v1')

        net0.load(checkpoint_path=model_agent0, optimizer=None, eligibility_traces=False)
        net1.load(checkpoint_path=model_agent1, optimizer=None, eligibility_traces=False)

        agents = {WHITE: TDAgent(WHITE, net=net1), BLACK: TDAgent(BLACK, net=net0)}

        evaluate_agents(agents, env, n_episodes)