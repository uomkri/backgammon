from new_nural_network.model import TDGammon
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from agents import TDAgent, TDAgent3ply, evaluate_agents

import gym

def start_evaluation():

    hidden_units = 160
    n_episodes = 100

    net0 = TDGammon(hidden_units=hidden_units, lr=0.1, lamda=None, init_weights=False)
    net1 = TDGammon(hidden_units=hidden_units, lr=0.1, lamda=None, init_weights=False)
    env = gym.make('gym_backgammon:backgammon-v1', game_type='khachapuri')

    path_for_better_agent = './saved/1000000.tar'
    path_for_weak_agent = './saved/1000000.tar'

    net0.load(checkpoint_path=path_for_better_agent, optimizer=None, eligibility_traces=False)
    net1.load(checkpoint_path=path_for_weak_agent, optimizer=None, eligibility_traces=False)

    agents = {WHITE: TDAgent(WHITE, net=net1), BLACK: TDAgent(BLACK, net=net0)}

    evaluate_agents(agents, env, n_episodes)

if __name__ == '__main__':
    start_evaluation()
