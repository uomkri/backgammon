import os
import gym
import pandas as pd
import sys
from itertools import count
import datetime

from agents import TDAgent, evaluate_agents
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from new_nural_network.model import TDGammon

import uuid


'''

Tables list:
1. game_id
2. step_number
3. qvalue
4. agent_color

1. game_id
2. winner_color

'''


def start_gen_data(game_type, games_count):
    env = gym.make('gym_backgammon:backgammon-v1', game_type=game_type)
    net = TDGammon(hidden_units=160, lr=0.1, lamda=None, init_weights=False, game_type=game_type)

    path_for_agent = 'saved/best_models/1000000.tar'

    net.load(checkpoint_path=path_for_agent, optimizer=None, eligibility_traces=False)

    df_steps_columns = ['game_id', 'step_number', 'qvalue', 'agent_color']
    df_steps = pd.DataFrame(columns=df_steps_columns)

    df_wins_columns = ['game_id', 'winner_color']
    df_wins = pd.DataFrame(columns=df_wins_columns)

    agents = {WHITE: TDAgent(WHITE, net=net, game_type=game_type), BLACK: TDAgent(BLACK, net=net, game_type=game_type)}

    wins = {WHITE: 0, BLACK: 0}

    for episode in range(games_count):
        game_id = uuid.uuid4()

        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        for step in count():
            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            # game_id, step_number, qvalue, color
            observation = env.game.board.get_board_features(env.current_agent)
            qvalue = net(observation)

            new_steps_df = pd.DataFrame(
                {'game_id': [game_id], 'step_number': [step], 'qvalue': [qvalue.detach()], 'agent_color': [agent.color]})
            df_steps = pd.concat([df_steps, new_steps_df], ignore_index=True, axis=0)


            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                # записываем во вторую таблицу
                new_wins_df = pd.DataFrame({'game_id': [game_id], 'winner_color': [agent.color]})
                df_wins = pd.concat([df_wins, new_wins_df], ignore_index=True, axis=0)

                if winner is not None:
                    wins[agent.color] += 1
                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print("EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%)".format(episode + 1, winner, step,
                    agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                    agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100))
                break

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next

    time_now = datetime.datetime.now()
    file_time = str(time_now.year) + '_' + str(time_now.month) + '_' + str(time_now.day) + '_' + str(time_now.hour) + '_' + str(time_now.minute)
    steps_df_save_path = f'games_data/{game_type}/steps_{games_count}_{file_time}'
    wins_df_save_path = f'games_data/{game_type}/wins_{games_count}_{file_time}'

    df_steps.to_csv(steps_df_save_path)
    df_wins.to_csv(wins_df_save_path)

    return wins

if __name__ == '__main__':
    games_count = 10000
    wins_count = start_gen_data('khachapuri', games_count)
    print(wins_count)