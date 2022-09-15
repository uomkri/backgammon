import gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK, COLORS, TOKEN

random.seed(0)
np.random.seed(0)


class RandomAgent:
    def __init__(self, color):
        self.color = color
        self.name = 'AgentExample({})'.format(self.color)

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (
        random.randint(1, 6), random.randint(1, 6))

    def choose_best_action(self, actions, env):
        return random.choice(list(actions)) if actions else None


def make_plays():
    env = gym.make('gym_backgammon:backgammon-v1', game_type='short')
    wins = {WHITE: 0, BLACK: 0}
    agents = {WHITE: RandomAgent(WHITE), BLACK: RandomAgent(BLACK)}
    games_count = 1000
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

            # print(
            #     "Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color], roll))

            actions = env.get_valid_actions(roll)
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
    make_plays()