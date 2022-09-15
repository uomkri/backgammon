from api.mongo_init import db
import uuid
import gym
from gym_backgammon.envs.utils.board import Board


class DBActions():
    def put_new_session(self, is_pvp):
        uid = str(uuid.uuid4())
        env = gym.make('gym_backgammon:backgammon-v1', game_type="khachapuri")
        agent_color, first_roll, observation = env.reset() 
        _board = env.game.board
        _off = env.game.board.off
        print(first_roll)
        print("ROLL1")
        print("START BOARD")
        env.game.board.render()
        print("VALID ACT")
        print(env.get_valid_actions(first_roll))

        entry = {
            "uid": uid,
            "board": _board,
            "off": _off,
            "curr_player": agent_color,
            "history": [],
            "multiplier": [1, None],
            "roll": list(first_roll),
            "is_pvp": is_pvp,
        }
        res = db.games.insert_one(entry)
        print(res)
        return [uid, env, is_pvp, agent_color, list(first_roll)]

    def get_database(self):
        res = list(db.games.find({}))
        for e in res:
            e['_id'] = str(e['_id'])
        print(res)
        return res


    def get_session(self, uid):
        res = db.games.find_one({"uid": uid})
        print(res)
        board = res["board"]
        off = res["off"]

        _board = Board('long')

        for i in range(len(board)):
            _board[i] = board[i]
        _board.off = off
        env = gym.make('gym_backgammon:backgammon-v1')
        env.reset()
        env.set_current_board(_board)
        env.set_current_agent(res["curr_player"]) 
        env.set_current_multiplier(res["multiplier"])

        print("GET BOARD")
        print(_board)

        print("GET SESSION")
        print(env.current_agent)
        print(env.get_valid_actions(res["roll"]))
        env.game.board.render()
        return [res, env]

    def set_roll(self, uid, roll):
        res = db.games.update_one({"uid": uid}, {
            "$set": {
                "roll": roll
            }
        })

    def update_recommended(self, uid, recommended):
        res = db.games.update_one({"uid": uid}, {
            "$set": {
                "recommended": recommended
            }
        })

    def update_game(self, uid: str, board, off, curr_player, moves, multiplier, roll, recommended):
        print(list(moves))
        res = db.games.update_one({"uid": uid}, {
            "$set": {
                "board": board,
                "off": off,
                "curr_player": curr_player,
                "history": moves,
                "multiplier": multiplier,
                "roll": roll,
                "recommended": recommended
            }
        })
        print(res)

    def update_multiplier(self, uid, multiplier):
        db.games.update_one({"uid": uid}, {
            "$set": {
                "multiplier": multiplier
            }
        })
        print("multiplier updated", uid, multiplier)
