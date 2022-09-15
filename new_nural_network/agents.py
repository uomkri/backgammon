import random
import time
from itertools import count
from random import randint, choice

import numpy as np

from gym_backgammon.envs.utils.game_utils import WHITE, BLACK, COLORS, get_opponent, get_winner


# AGENT ============================================================================================


class Agent:
    def __init__(self, color, game_type):
        self.color = color
        self.name = 'Agent({})'.format(COLORS[color])
        self.game_type = game_type

    def roll_dice(self):
        if self.game_type == 'short':
            return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (
                random.randint(1, 6), random.randint(1, 6))
        while True:
            roll = [randint(1, 6), randint(1, 6)]
            if roll[1] == roll[0] and roll[0] == 1:
                continue
            roll = (roll[0], roll[0], roll[0], roll[0]) if roll[0] == roll[1] else roll
            return roll

    def choose_best_action(self, actions, env):
        raise NotImplementedError

    def set_new_color(self, color):
        self.color = color


# RANDOM AGENT =======================================================================================


class RandomAgent(Agent):
    def __init__(self, color, game_type):
        super().__init__(color, game_type)
        self.name = 'RandomAgent({})'.format(COLORS[color])
        self.game_type = game_type

    def choose_best_action(self, actions, env):
        return choice(list(actions)) if actions else None


# TD-GAMMON AGENT =====================================================================================


class TDAgent(Agent):
    def __init__(self, color, net, game_type):
        super().__init__(color, game_type)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0

            for i, action in enumerate(actions):
                if env.game_type == 'short':
                    old_state = env.game.save_state()
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)
                if env.game_type == 'short':
                    env.game.restore_state(old_state)
                else:
                    env.game.restore_state(env.current_agent, action)

            new_values = []
            for i in values:
                new_values += i.detach()

            values = new_values

            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action

    def get_recommendations(self, actions, env):
        recommendations = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0

            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = (self.net(observation), action)

                env.game.restore_state(env.current_agent, action)

            new_values = []
            for i in values:
                new_values.append((round(float(i[0].detach()), 5), i[1]))

            recommendations = sorted(new_values, key=lambda e: e[0], reverse=True)
        return recommendations

    def who_is_actual_winner(self, env):
        current_agent = env.current_agent
        white_agent_rec = self.net(env.game.board.get_board_features(current_agent))

        env.game.board.make_desk_mirror()
        black_agent_rec = self.net(env.game.board.get_board_features(get_opponent(current_agent)))
        env.game.board.make_desk_mirror()

        return round(float(white_agent_rec), 5), round(float(black_agent_rec),5)

    def should_double_recommendation(self, env):
        white_agent_rec, black_agent_rec = self.who_is_actual_winner(env=env)
        actual_winner = WHITE if white_agent_rec > black_agent_rec else BLACK

        if self.color == actual_winner:
            return True
        else:
            return False

class TDAgent3ply(Agent):
    def __init__(self, color, net):
        super().__init__(color)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.choose_2ply_best_action(env)
                # values[i] = self.net(observation)

                # restore the board and other variables (undo the action)
                env.game.restore_state(env.current_agent, action)

            new_values = []
            for i in values:
                new_values += i.detach()
            values = new_values

            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action

    def choose_2ply_best_action(self, env):
        all_rolls = [(a,b) for a in range(1,7) for b in range(a,7)]
        value = 0
        for roll in all_rolls:
            if roll[0] == roll[1] and roll[0] == 1:
                continue
            if roll[0] == roll[1]:
                roll = (roll[0], roll[0], roll[0], roll[0])
            probability = 1/18 if roll[0] != roll[1] else 1/36
            valid_actions = env.get_valid_actions(roll)
            min_value = 1
            for i, action in enumerate(valid_actions):
                observation, reward, done, info = env.step(action)
                new_value = self.net(observation)
                env.game.restore_state(env.current_agent, action)
                if new_value < min_value:
                    min_value = new_value
            value += probability * min_value

        return value

    def choose_3ply_best_action(self, env):
        pass

def evaluate_agents(agents, env, n_episodes):
    wins = {WHITE: 0, BLACK: 0}

    for episode in range(n_episodes):
        print('episode: ', episode)

        agent_color, first_roll, observation = env.reset()
        agent = agents[agent_color]

        t = time.time()

        for i in count():

            if first_roll:
                roll = first_roll
                first_roll = None
            else:
                roll = agent.roll_dice()

            actions = env.get_valid_actions(roll)
            action = agent.choose_best_action(actions, env)
            observation_next, reward, done, winner = env.step(action)

            if done:
                if winner is not None:
                    wins[agent.color] += 1
                tot = wins[WHITE] + wins[BLACK]
                tot = tot if tot > 0 else 1

                print("EVAL => Game={:<6d} | Winner={} | after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(episode + 1, winner, i,
                    agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
                    agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
                break

            agent_color = env.get_opponent_agent()
            agent = agents[agent_color]

            observation = observation_next
    return wins