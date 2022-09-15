from operator import itemgetter
from flask.json import jsonify
from api.utils.utils import Utils
from api.actions.db_actions import DBActions
import random
from api.flask_init import app
from flask import request, jsonify
import numpy as np
import torch.nn.functional as F
import time
from new_nural_network.agents import TDAgent
from new_nural_network.model import TDGammon
from gym_backgammon.envs.utils.game_utils import BLACK, WHITE, get_roll, get_winner
import multiprocessing as mp
from catboost import CatBoostClassifier




class ApiActions():
    def __init__(self):
        self.db = DBActions()
        self.utils = Utils()
        self.app = app
        self.pool = mp.Pool()

        low = np.zeros((196, 1))
        high = np.ones((196, 1))

        for i in range(3, 97, 4):
            high[i] = 6.0
        high[96] = 7.5

        for i in range(101, 195, 4):
            high[i] = 6.0
        high[194] = 7.5

        self.model = CatBoostClassifier()
        self.model.load_model('../../cube_model/catboost_model')


        # self.observation_space = Box(low=low, high=high)    
        # self.white_agent = DQNAgent(color=0, state_shape=self.observation_space.shape, hidden_size=[100, 10], roll=[])
        # self.black_agent = DQNAgent(color=1, state_shape=self.observation_space.shape, hidden_size=[100, 10], roll=[])
        # self.white_agent.load_state_dict(torch.load("neural_network/player_1_10k_2.pth"))
        # self.black_agent.load_state_dict(torch.load("neural_network/player_1_10k_2.pth"))

        self.net = TDGammon(hidden_units=160, lr=0.1, lamda=None, init_weights=False, game_type="khachapuri")
        self.net.load(checkpoint_path="./new_nural_network/saved/1000000.tar", optimizer=None, eligibility_traces=False)
        self.black_agent=TDAgent(BLACK, net=self.net, game_type="khachapuri")
        self.white_agent=TDAgent(WHITE, net=self.net, game_type="khachapuri")

        


    def calculate_probability(self, qvalue):
        m = F.softmax(dim=0, input=qvalue)
        return m


    # true - favorable for white, false - favorable for black
    def calculate_doubling_offer(self, env):
        # sum_b = 0
        # sum_w = 0
        # observation_w = torch.tensor(game.get_board_features(0))
        # observation_b = torch.tensor(game.get_board_features(1))

        # combs = get_all_dices_combs()

        # ind = random.randint(0, len(combs)-1)

        # in_line = game.moves_in_line(game.get_valid_plays(0, combs[ind], game.board))
        # best_action = 0
        # qvalues = self.white_agent.forward(observation_w, in_line)
        # if qvalues is not None:
        #     max = -10000
        #     index = -1
        #     for i in range(len(qvalues)):
        #         if qvalues[i] > max:
        #             max = qvalues[i]
        #             index = i
        #             best_action = max.item()
        #         else:
        #             best_action = 0
        #     sum_w += best_action

        # ind = random.randint(0, len(combs)-1)

        # in_line = game.moves_in_line(game.get_valid_plays(1, combs[ind], game.board))
        # best_action = 0
        # qvalues = self.black_agent.forward(observation_b, in_line)
        # if qvalues is not None:
        #     max = -10000
        #     index = -1
        #     for i in range(len(qvalues)):
        #         if qvalues[i] > max:
        #             max = qvalues[i]
        #             index = i
        #             best_action = max.item()
        #         else:
        #             best_action = 0
        #     sum_b += best_action
        # if sum_w > sum_b:
        #     return True
        # else:
        #     return False

        white, black = env.should_double(self.black_agent)

        if (black > white):
            return False
        else:
            return True

    def move_agent(self, uid: str):
        res = self.db.get_session(uid)

        dec = random.randint(0, 2)

        env = res[1]
        _res = res[0]
        _player = BLACK
        history = _res["history"]
        roll = list(self.black_agent.roll_dice())
        multiplier = _res["multiplier"]
        _board = env.game.board
        # _roll = self.black_agent.roll_dice()
        _moves = env.get_valid_actions(roll)
        # in_line = game.moves_in_line(_moves)
        _action = self.black_agent.choose_best_action(_moves, env)
        print("VALIDNIE HODI")
        print(_moves)
        print("VIBRANNY ACTION")
        print(_action)
        print("ROLL")
        print(roll)
        print(env.current_agent)
        print("NIGGA")
        # observation = torch.tensor(game.get_board_features(_player))
        best_action = [[]]

        # qvalues = self.black_agent.forward(observation, in_line)
        # if qvalues is not None:
        #     max = -10000
        #     index = -1
        #     for i in range(len(qvalues)):
        #         if qvalues[i] > max:
        #             max = qvalues[i]
        #             index = i
        #     best_action = in_line[index]
        # else:
        #     best_action = [[]]
        dec = 0

        white, black = env.should_double(self.black_agent)

        if (black > white):
            rnd = random.randint(0, 1)
            if (rnd == 1):
                dec = 1
        else:
            dec = 0

        dec = 0

        if dec == 1 and (multiplier[1] == 1 or multiplier[1] == None) and multiplier[0] < 64:            
            return True
        else:
            # index = random.randint(0, len(in_line) - 1)
            # selected_moves = in_line[index]

            env.game.board.render()
            print("BEFORE")

            last_turn = 0
            
            if len(history) > 0:
                last_turn = history[-1]["turn_number"]

            print("best action", best_action)
            valid_actions = env.get_valid_actions(roll)


            recommended = self.black_agent.get_recommendations(valid_actions, env)
            first_recommended = []
            if (len(recommended) > 0):
                first_recommended = recommended[0]

            rec = {
                "turn_number": last_turn+1,
                "player": self.utils.map_player_to_front(1),
                "moves": [],
                "recommended": recommended,
                "first_recommended": first_recommended,
                "timestamp": time.time(),
                "dice": roll
            }
            if len(_action) == 0:
                rec["moves"].append({
                    "dice": roll,
                    "from_position": None,
                    "to_position": None,
                    "player": self.utils.map_player_to_front(1)
                })
            else:
                for e in _action:
                    rec["moves"].append({
                        "from_position": e[0],
                        "to_position": e[1],
                        "player": self.utils.map_player_to_front(1),
                        # GET NEW ROLL
                        "dice": roll
                    })
        
  
           # game.execute_play(_player, best_action)
            observation_next, reward, done, winner = env.step(_action)

            history.append(rec)
            new_board = env.game.board
            # next_player = WHITE
            new_off = env.game.board.off

            env.game.board.render()

            self.db.update_game(
                uid=uid,
                board=new_board,
                curr_player=env.get_opponent_agent(),
                moves=history,
                off=new_off,
                multiplier=multiplier,
                roll=roll,
                recommended=[]
            )
            return False


    def start(self, name, is_pvp):
        if name == "khachapuri":
            res = self.db.put_new_session(is_pvp)
            uid = res[0]
            env = res[1]
            history = []

            if is_pvp == False:
                first = res[3]
                if first == BLACK:
                    self.move_agent(uid)

            game_res = self.db.get_session(uid)
            env = game_res[1]
            _res = game_res[0]
            _player = _res["curr_player"]
            history = _res["history"]
            multiplier = _res["multiplier"]


            current_roll = list(self.black_agent.roll_dice())
            self.db.set_roll(uid=uid, roll=current_roll)
            board = env.game.board
            valid_moves = env.game.get_valid_plays(
                check_player=0,
                roll=current_roll,
                #roll=[6, 6, 6, 6],
                board=board
            )

            in_line = env.game.moves_in_line(valid_moves)

            should_double = False
            sd_roll = self.white_agent.roll_dice()[0]

            if (sd_roll > 3):
                should_double = self.white_agent.should_double_recommendation(env)

            valid_actions = env.get_valid_actions(current_roll)

            recommended = self.white_agent.get_recommendations(valid_actions, env)

            print('recommended: ', recommended)

            self.db.update_recommended(uid, recommended)

            print("START")
            res = self.utils.return_board(
                _id=uid,
                _player=0,
                _roll=current_roll,
                #_roll=[6, 6, 6, 6],
                _moves=valid_moves,
                _hist=sorted(history, key=itemgetter('turn_number')),
                _board=board,
                _multiplier=[1, None],
                _doubling_offered=False,
                _is_finished=False,
                _winner=None,
                _suggested=recommended,
                _doubling_suggestion=should_double,
                _is_pvp=is_pvp
            )
            print(res)
            return res
        else:
            return jsonify(
                message="Unknown type"
            ).headers.add('Access-Control-Allow-Origin', "*")
    
    def move(self, uid):
        print(uid)

        _uid = str(uid)

        req = request.get_json()

        mapped = map(lambda e: self.utils.map_move_for_execution(e), req)

        game_res = self.db.get_session(uid)
        env = game_res[1]
        _res = game_res[0]
        _player = _res["curr_player"]
        history = _res["history"]
        roll = _res["roll"]
        multiplier = _res["multiplier"]
        print("REQ", req)


        last_turn = 0
            
        if len(history) > 0:
            last_turn = history[-1]["turn_number"]

        recommended = _res["recommended"]

        first_recommended = []

        if (len(recommended) > 0):
            first_recommended = recommended[0]

        rec = {
            "turn_number": last_turn+1,
            "moves": [],
            "player": self.utils.map_player_to_front(_player),
            "recommended": recommended,
            "first_recommended": first_recommended,
            "timestamp": time.time(),
            "dice": roll

        }

        if len(req) == 0:
            rec["moves"].append({
                "player": self.utils.map_player_to_front(_player),
                "from_position": None,
                "to_position": None,    
                "dice": roll                
            })
        else:
            for e in req:
                    rec["moves"].append({
                       "player": self.utils.map_player_to_front(_player),
                        "from_position": e["from_position"],
                        "to_position": e["to_position"],
                        "dice": roll 
                    })
        

        # game.execute_play(_player, mapped)
        _, _, done, winner = env.step(mapped)

        history.append(rec)

        next_player = 0
        if _player == 0:
            next_player = 1

        self.db.update_game(
            uid=uid,
            board=env.game.board,
            curr_player=env.get_opponent_agent(),
            off=env.game.board.off,
            moves=history,
            roll=roll,
            multiplier=multiplier,
            recommended=recommended
        )
        dec = None
        is_finished = False
        print(env.game.board.off)
        if winner is not None:
            print('У НАС ЕСТЬ ПОБЕДИТЕЛЬ!!!', winner)
            is_finished = True
            env.game.board.render()
        elif _res["is_pvp"] == False:
            dec = self.move_agent(_uid)


        game_res = self.db.get_session(uid)
        env = game_res[1]
        _res = game_res[0]
        _player = _res["curr_player"]
        history = _res["history"]
        multiplier = _res["multiplier"]


        _roll = get_roll()
        self.db.set_roll(uid, _roll)

        moves = env.game.get_valid_plays(_player, _roll, env.game.board)
        in_line = env.game.moves_in_line(moves)

        valid_actions = env.get_valid_actions(_roll)


        recommended = self.white_agent.get_recommendations(valid_actions, env)
        # recommended.append({
        #     "qvalue": 1,
        #     "move": env.get_recommendations(self.black_agent, moves)
        # })
        self.db.update_recommended(uid, recommended)

        # observation = torch.tensor(game.get_board_features(_player))
        # best_actions = []
        # recommended = []

        # qvalues = self.white_agent.forward(observation, in_line)
        # if qvalues is not None:
        #     max = -10000
        #     index = -1
        #     for i in range(len(qvalues)):
        #         best_actions.append({
        #             "qvalue": qvalues[i].item(),
        #             "move": in_line[i]
        #         })
        #     recommended = self.utils.select_best_actions(best_actions)
        # else:
        #     best_actions = []
            
        # print("BEST", best_actions)
        # print("SUGGESTED", recommended)

        print("AFTER BOT", env.game.board.off)
        winner = get_winner(env.game.board.off)
        if winner is not None:
            print('У НАС ЕСТЬ ПОБЕДИТЕЛЬ!!!', winner)
            is_finished = True
            env.game.board.render()

        should_double = False
        sd_roll = self.white_agent.roll_dice()[0]

        if (sd_roll > 3):
            should_double = self.white_agent.should_double_recommendation(env)

        try:
            res = self.utils.return_board(
                _id=_uid,
                _player=_player,
                _roll=_roll,
                _moves=moves,
                _hist=sorted(history, key=itemgetter('turn_number')),
                _board=env.game.board,
                _multiplier=multiplier,
                _doubling_offered=dec,
                _is_finished=is_finished,
                _winner=self.utils.map_player_to_front(winner),
                _suggested=recommended,
                _doubling_suggestion=should_double,
                _is_pvp=_res["is_pvp"]
            )  # .headers.add('Access-Control-Allow-Origin', "*")
            return res
        except Exception as e:
            return ""

    def get_win_eval(self, uid):
        res = self.db.get_session(uid)
        env = res[1]

        no_double, double_pass, double_take = env.get_cube_recommendations(model=self.model, white_agent=self.white_agent)


        eval = self.white_agent.who_is_actual_winner(env)
        return jsonify(
            eval=eval,
            no_double=no_double,
            double_pass=double_pass,
            double_tak=double_take
        )


    def double_offered(self, uid):
        res = self.db.get_session(uid)
        env = res[1]
        res = res[0]

        should_double = True
        sd_roll = self.black_agent.roll_dice()[0]

        if (sd_roll > 3):
           should_double = self.black_agent.should_double_recommendation(env)
        
        decision = should_double

        
        game_res = self.db.get_session(uid)
        game = game_res[1]
        _res = game_res[0]
        multiplier = _res["multiplier"]
        if multiplier[0] == 64:
            return jsonify(
                error="Multiplier already maxed out",
                decision=None
            )
        else:
            if decision == True:
                new_mult = [multiplier[0]*2, 1]
                self.db.update_multiplier(uid, new_mult)
        return jsonify(decision=decision)
            
    
    def doubling_agreement(self, uid, answer):
        game_res = self.db.get_session(uid)
        env = game_res[1]
        _res = game_res[0]
        _player = _res["curr_player"]
        history = _res["history"]
        roll = _res["roll"]
        multiplier = _res["multiplier"]     

        print(answer)   

        if answer == "true":
            dec = self.move_agent(uid)
            game_res = self.db.get_session(uid)
            env = game_res[1]
            _res = game_res[0]
            _player = _res["curr_player"]
            history = _res["history"]
            roll = self.black_agent.roll_dice()
            self.db.update_multiplier(uid, roll)
            multiplier = _res["multiplier"]

            moves = env.game.get_valid_plays(0, roll, env.game.board)

            # observation = torch.tensor(game.get_board_features(0))
            # best_actions = []
            # recommended = []

            # qvalues = self.white_agent.forward(observation, in_line)
            # if qvalues is not None:
            #     max = -10000
            #     index = -1
            #     for i in range(len(qvalues)):
            #         best_actions.append({
            #             "qvalue": qvalues[i].item(),
            #             "move": in_line[i]
            #         })
            #     recommended = self.utils.select_best_actions(best_actions)
            # else:
            #     best_actions = []

            new_mult = [multiplier[0]*2, 0]
            self.db.update_multiplier(uid, new_mult)
            res = self.utils.return_board(
                _id=uid,
                _player=0,
                _roll=roll,
                _moves=moves,
                _hist=sorted(history, key=itemgetter('turn_number')),
                _board=env.game.board,
                _multiplier=new_mult,
                _doubling_offered=dec,
                _is_finished=False,
                _winner=None,
                _suggested=[],
                _doubling_suggestion=self.calculate_doubling_offer(env),
                _is_pvp=False
                )
            return res
        else:
            res = self.utils.return_board(
                _id=uid,
                _player=0,
                _roll=roll,
                _moves=env.game.get_valid_plays(0, roll, env.game.board),
                _hist=sorted(history, key=itemgetter('turn_number')),
                _board=env.game.board,
                _multiplier=multiplier,
                _doubling_offered=False,
                _is_finished=True,
                _winner=1,
                _doubling_suggestion=False,
                _is_pvp=False
            )
            return res
    

    def update_multiplier(self, uid, mult):
        self.db.update_multiplier(uid, mult)

    def get_win_probs(self, uid):
        game_res = self.db.get_session(uid)
        env = game_res[1]

        probs = env.get_wins_probabilities(self.pool, self.white_agent)

        print("probabilities for WHITE", probs)

        return { "distribution": probs }
    


