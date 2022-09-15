from new_nural_network.agents import TDAgent
from new_nural_network.model import TDGammon
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK, NUM_POINTS
import gym


def test_agent():

    big_weight_path = './saved/long/450000.tar'
    model = TDGammon(hidden_units=160, lr=0.1, lamda=None, init_weights=False)
    model.load(checkpoint_path=big_weight_path, optimizer=None, eligibility_traces=False)
    agent = TDAgent(WHITE, net=model)

    env = gym.make('gym_backgammon:backgammon-v1', game_type='long')
    env.reset()
    env.set_current_agent(BLACK)

    env.game.board[12] = (14, BLACK)
    env.game.board[13] = (1, BLACK)
    roll = [5,1]

    valid_actions = env.get_valid_actions(roll)

    print('valid_actions: ', valid_actions)
    print('current_player: ', env.current_agent)
    recommendations = agent.get_recommendations(actions=valid_actions, env=env)
    print('recommendations:', recommendations)
    print('Оценка ситуации(белый и черный): ', agent.who_is_actual_winner(env=env))
    print('Доска')
    env.game.board.render()


def init_board():
    board = [(0, None)] * NUM_POINTS
    board[0] = (3, WHITE)
    board[8] = (1, WHITE)
    board[18] = (11, WHITE)
    #
    board[12] = (4, BLACK)
    #board[14] = (2, BLACK)

    #board[12] = (4, BLACK)

    board[6] = (11, BLACK)

    return board


if __name__ == '__main__':
    test_agent()