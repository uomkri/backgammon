import gym
import time
from itertools import count
import random
import numpy as np
import pandas as pd
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK, COLORS, TOKEN
from new_nural_network.agents import TDAgent, RandomAgent
from new_nural_network.model import TDGammon
import uuid
from catboost import CatBoostClassifier


def make_plays(games_count=10, game_type='khachapuri'):
    env = gym.make('gym_backgammon:backgammon-v1', game_type=game_type)

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
    mars_wins = {WHITE: 0, BLACK: 0}
    columns_names = [x for x in range(196)] # кол-во фичей в представлении доски хачапури
    columns_names.append('game_id')
    columns_names.append('target')

    khachapuri_games_df = pd.DataFrame(columns=columns_names)

    model = CatBoostClassifier()
    model.load_model('../cube_model/catboost_model')

    for games_iter in range(games_count):
        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]
        t = time.time()
        new_game_observations = []
        game_id = uuid.uuid4()

        for i in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            if agent_color == WHITE:
                no_double, double_pass, double_take = env.get_cube_recommendations(model=model, white_agent=agents[WHITE])
                print(no_double, double_pass, double_take)

            new_game_observations.append(observation)

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                if winner is not None:
                    wins[winner] += 1

                    white_mars_win = winner == WHITE and env.game.board.off[BLACK] == 0
                    for observation in new_game_observations:
                        observation.append(game_id)
                        observation.append(white_mars_win)

                    if winner == WHITE:
                        if env.game.board.off[BLACK] == 0:
                            mars_wins[WHITE] += 1
                    else:
                        if env.game.board.off[WHITE] == 0:
                            mars_wins[BLACK] += 1

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

        new_game_df = pd.DataFrame(new_game_observations, columns=columns_names)
        khachapuri_games_df = pd.concat([khachapuri_games_df, new_game_df], ignore_index=True)

    print(f'For {games_count} games mars wins: white - {mars_wins[WHITE]}, black = {mars_wins[BLACK]}')

    return khachapuri_games_df


if __name__ == '__main__':
    rules = 'khachapuri'
    games_count = 1
    khachapuri_games_df = make_plays(games_count, rules)
    t = time.localtime()
    # observations_save_dir = f'./games_data/khachapuri/{rules}_{games_count}_{t.tm_year}_{t.tm_mon}_{t.tm_mday}_{t.tm_hour}_{t.tm_min}.csv'
    # khachapuri_games_df.to_csv(observations_save_dir)

    # print(khachapuri_games_df)
