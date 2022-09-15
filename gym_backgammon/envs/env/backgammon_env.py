import gym
from gym.spaces import Box
from gym_backgammon.envs.rules.khachapuri import KhachapuriBackgammon
from gym_backgammon.envs.rules.long import LongBackgammon
from gym_backgammon.envs.rules.short import ShortBackgammon
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from random import randint
import numpy as np
from gym_backgammon.envs.utils.game_utils import get_opponent, get_winner, get_all_dices_combs


class BackgammonEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array', 'state_pixels']}

    def __init__(self, game_type='khachapuri'):
        self.game_type = game_type
        if self.game_type == 'long':
            self.game = LongBackgammon()
        elif self.game_type == 'khachapuri':
            self.game = KhachapuriBackgammon()
        elif self.game_type == 'short':
            self.game = ShortBackgammon()
        else:
            print('неверный тип игры, инициализирую хачапури')
            self.game = KhachapuriBackgammon()

        self.current_agent = WHITE
        self.reward_for_win = 15
        self.taxes_mode = False
        self.__global_taxes = 0

        if self.game_type == 'short':
            low = np.zeros((198, 1))
            high = np.ones((198, 1))
        else:
            low = np.zeros((196, 1))
            high = np.ones((196, 1))

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        self.observation_space = Box(low=low, high=high)
        self.counter = 0
        self.max_length_episode = 1000000
        self.viewer = None

    def get_all_observations(self, valid_actions) -> list:
        assert self.game_type != 'short', f'Not implemented function for {self.game_type} game type'
        observations = []
        for action in valid_actions:
            self.game.execute_play(self.current_agent, action)
            new_observation = self.game.board.get_board_features(self.current_agent)
            observations.append(new_observation)

            self.game.restore_state(self.current_agent, action)

        return observations

    def activate_taxes_mode(self):
        self.taxes_mode = True

    def get_recommendations(self, agent, valid_actions):
        assert self.game_type != 'short', f'Not implemented function for this: {self.game_type} game type'
        agent.set_new_color(WHITE)
        best_action = agent.choose_best_action(actions=valid_actions, env=self)
        agent.set_new_color(BLACK)

        return best_action

    # only for white agent
    def get_cube_recommendations(self, model, white_agent):
        observation_white = self.game.board.get_board_features(WHITE)

        self.game.board.make_desk_mirror()
        observation_black_as_white = self.game.board.get_board_features(WHITE)
        self.game.board.make_desk_mirror()

        N = self.game.multiplier[0]
        win_prob = white_agent.net(observation_white).item()
        mars_win_prob = model.predict_proba(observation_white)[1]

        lose_prob = 1 - win_prob
        mars_lose_prob = model.predict_proba(observation_black_as_white)[1]

        double_pass = round(N * -1,
                            3)
        double_take = round(win_prob * (2*N) +
                       mars_win_prob * (4*N) +
                       lose_prob * (2*N*-1) +
                       mars_lose_prob * (4*N*-1),
                            3)
        no_double = round(win_prob * N +
                     mars_win_prob * (2*N) +
                     lose_prob * (N*-1) +
                     mars_lose_prob * (2*N*-1),
                          3)

        return no_double, double_pass, double_take

    def should_double(self, agent):
        assert self.game_type != 'short', f'Not implemented function for this: {self.game_type} game type'
        agent_for_features = WHITE
        agent.set_new_color(agent_for_features)

        white = agent.net(self.game.board.get_board_features(agent_for_features))
        self.game.board.make_desk_mirror()

        black = agent.net(self.game.board.get_board_features(agent_for_features))
        self.game.board.make_desk_mirror()

        agent.set_new_color(BLACK)

        return white, black

    def set_short_rule_params(self, board, off, bar):
        self.game.board = board
        self.game.off = off
        self.game.bar = bar

    def set_current_agent(self, color):
        self.current_agent = color

    def set_current_board(self, board):
        self.game.board = board

    def set_current_multiplier(self, multiplier):
        self.game.multiplier = multiplier

    def set_current_off(self, off):
        self.game.board.off = off

    def deactivate_taxes_mode(self):
        self.taxes_mode = False

    def set_reward_for_win(self, reward):
        self.reward_for_win = reward

    def step(self, action):
        self.game.execute_play(self.current_agent, action)
        if self.game_type == 'short':
            observation = self.game.get_board_features(get_opponent(self.current_agent))
        else:
            observation = self.game.board.get_board_features(get_opponent(self.current_agent))

        if self.game_type == 'short':
            winner = get_winner(self.game.off)
        else:
            winner = get_winner(self.game.board.off)

        reward = 0
        done = False

        if winner is not None or self.counter > self.max_length_episode:
            if winner == WHITE:
                reward = 1
            done = True

        self.counter += 1

        return observation, reward, done, winner

    def reset(self):
        roll = randint(1, 6), randint(1, 6)

        while roll[0] == roll[1]:
            roll = randint(1, 6), randint(1, 6)

        if roll[0] > roll[1]:
            self.current_agent = WHITE
            if self.game_type == 'short':
                roll = (-roll[0], -roll[1])
        else:
            self.current_agent = BLACK

        if self.game_type == 'long':
            self.game = LongBackgammon()
        elif self.game_type == 'khachapuri':
            self.game = KhachapuriBackgammon()
        elif self.game_type == 'short':
            self.game = ShortBackgammon()
        else:
            print('неверный тип игры, инициализирую хачапури')
            self.game = KhachapuriBackgammon()

        self.counter = 0

        if self.game_type == 'short':
            return self.current_agent, roll, self.game.get_board_features(self.current_agent)

        return self.current_agent, roll, self.game.board.get_board_features(self.current_agent)

    def get_valid_actions(self, roll):
        if self.game_type == 'short':
            return self.game.get_valid_plays(self.current_agent, roll)

        actions = self.game.get_valid_plays(self.current_agent, roll, self.game.board)
        actions_in_line = self.game.moves_in_line(actions)
        if len(actions_in_line) > 0:
            return actions_in_line
        return None

    # def get_mirror(self):
    #     self.game.make_desk_mirror()

    def get_opponent_agent(self):
        self.current_agent = get_opponent(self.current_agent)
        return self.current_agent

    def get_wins_probabilities(self, processes_pool, agent):
        all_dices = get_all_dices_combs()
        all_dices_plus_env = []

        for dice in all_dices:
            new_dice = []
            if agent.color == WHITE and self.game_type == 'short':
                for j in range(len(dice)):
                    new_dice.append(-dice[j])
            else:
                new_dice = dice
            all_dices_plus_env.append({'env': self, 'roll': new_dice, 'agent': agent})

        result = processes_pool.map(self.get_all_best_actions, all_dices_plus_env)
        q_value_array = []
        q_value_probs = {}
        for i in range(11):
            q_value_probs[i/10.0] = 0

        for j in result:
            roll_0 = j['roll'][0]
            roll_1 = j['roll'][1]
            q_value = j['q_value'].detach()[0]
            probability = 1 / 18 if roll_0 != roll_1 else 1 / 36
            q_value_array.append(round(float(q_value), 2))
            q_value_probs[round(float(q_value), 1)] += probability

        prob_sum = 0.0
        for el in q_value_probs:
            q_value_probs[el] = round(q_value_probs[el], 1)
            prob_sum += q_value_probs[el]
        q_value_array.sort()
        return q_value_probs

    def get_all_best_actions(self, obj):
        env = obj['env']
        agent = obj['agent']
        roll = obj['roll']

        actions = env.get_valid_actions(roll)
        action = agent.choose_best_action(actions, env=env)

        if env.game_type == 'short':
            old_state = env.game.save_state()
            observation, reward, done, info = env.step(action)
            q_value = agent.net(observation)
            env.game.restore_state(old_state)
        else:
            observation, reward, done, info = env.step(action)
            q_value = agent.net(observation)
            env.game.restore_state(env.current_agent, action)

        return {'action': action, 'roll': roll, 'q_value': q_value.detach()}


