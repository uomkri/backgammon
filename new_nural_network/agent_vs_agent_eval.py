from new_nural_network.evaluate_config import EvaluateConfig
from new_nural_network.utils import args_evaluate
from new_nural_network.model import TDGammon, TDAgent
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from new_nural_network.agents import evaluate_agents
import gym


def start_evaluation():
    args_for_eval = EvaluateConfig(episodes=50,
                                   hidden_units_agent0=40,
                                   hidden_units_agent1=40,
                                   model_type="nn",
                                   model_agent1="./new_nural_network/saved/exp1_20211210_0116_38_288759_1000.tar",
                                   model_agent0="./new_nural_network/saved/exp1_20211210_0116_38_288759_1000.tar"
                                   )
    args_evaluate(args_for_eval)




if __name__ == "__main__":
    env = gym.make('gym_backgammon:backgammon-v1')
    args = EvaluateConfig(episodes=50,
                                   hidden_units_agent0=80,
                                   hidden_units_agent1=80,
                                   model_type="nn",
                                   model_agent1="./new_nural_network/saved/best_models/exp1_20211210_1435_28_989201_10000.tar",
                                   model_agent0="./new_nural_network/saved/best_models/190000.tar"
                                   )

    net0 = TDGammon(hidden_units=args.hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
    net1 = TDGammon(hidden_units=args.hidden_units_agent1, lr=0.1, lamda=None, init_weights=False)

    net0.load(checkpoint_path=args.model_agent0, optimizer=None, eligibility_traces=False)
    net1.load(checkpoint_path=args.model_agent1, optimizer=None, eligibility_traces=False)

    agents = {WHITE: TDAgent(WHITE, net=net1), BLACK: TDAgent(BLACK, net=net0)}
    evaluate_agents(agents, env, 50)



