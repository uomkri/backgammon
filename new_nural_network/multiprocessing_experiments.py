import multiprocessing as mp
import gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK, COLORS, TOKEN
from gym_backgammon.envs.utils.game_utils import get_all_dices_combs
from new_nural_network.agents import TDAgent
from new_nural_network.model import TDGammon


# def get_all_best_actions(obj):
#     env = obj['env']
#     agent = obj['agent']
#     roll = obj['roll']
#
#     actions = env.get_valid_actions(roll)
#     action = agent.choose_best_action(actions, env=env)
#
#     if env.game_type == 'short':
#         old_state = env.game.save_state()
#         observation, reward, done, info = env.step(action)
#         q_value = agent.net(observation)
#         env.game.restore_state(old_state)
#     else:
#         observation, reward, done, info = env.step(action)
#         q_value = agent.net(observation)
#         env.game.restore_state(env.current_agent, action)
#
#     return {'action': action, 'roll': roll, 'q_value': q_value.detach()}


def make_plays(games_count=10, game_type='khachapuri'):
    env = gym.make('gym_backgammon:backgammon-v1', game_type=game_type)

    # pool = mp.Pool()

    if game_type == 'khachapuri':
        big_weight_path = './saved/best_models/1000000.tar'
    elif game_type == 'short':
        big_weight_path = './saved/short/short__20220524_1234_00_818269_1520000.tar'
    else:
        assert 'wtf'

    model = TDGammon(hidden_units=160, lr=0.1, lamda=None, init_weights=False, game_type=game_type)
    model.load(checkpoint_path=big_weight_path, optimizer=None, eligibility_traces=False)
    agents = {WHITE: TDAgent(color=WHITE, net=model, game_type=game_type),
              BLACK: TDAgent(color=BLACK, net=model, game_type=game_type)}

    wins = {WHITE: 0, BLACK: 0}

    showed_actions = False

    #
    # if game_type == 'short':
    #     env.game.render()
    #     print('short!')
    #     print(env.game.board)
    # else:
    #     print('not short')
    #     env.game.board.render()
    #     print(env.game.board)

    for games_iter in range(games_count):
        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]
        t = time.time()

        for i in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            # for dice in all_dices:
            #     new_dice = []
            #     if agent.color == WHITE:
            #         for j in range(len(dice)):
            #             new_dice.append(-dice[j])
            #     else:
            #         new_dice = dice
            #     actions = env.get_valid_actions(new_dice)
            #     action = agent.choose_best_action(actions, env=env)

            q_value_probs = env.get_wins_probabilities(pool, agent)
            # print('actual agent: ', agent.color)
            # print('q_value_probs: ', q_value_probs)
            # print('------')

            # print(
            #     "Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color], roll))

            actions = env.get_valid_actions(roll)
            print(actions)
            env.game.render()
            if not showed_actions:
                print(actions)
                showed_actions = True
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                if winner is not None:
                    wins[winner] += 1

                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print(
                    "Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
                        games_iter, winner, i,
                        agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                        agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
                break

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]
            observation = observation_next


if __name__ == '__main__':
    t = time.time()
    games_count = 1
    game_type = 'short'
    make_plays(games_count=games_count, game_type=game_type)
    print(f'{games_count} plays time: ', time.time() - t)
    # game_type = 'khachapuri'
    # make_plays(games_count=games_count, game_type=game_type)

