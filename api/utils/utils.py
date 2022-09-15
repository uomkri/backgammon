from array import array
import flask
from operator import itemgetter


class Utils():
    def map_player_to_front(self, p):
        if p == 0:
            return 2
        elif p == 1:
            return p

    def select_best_actions(self, actions):
        sort = sorted(actions, key=itemgetter('qvalue'), reverse=True)
        if len(sort) > 5:
            return sort[:5]
        else:
            return sort

    def map_move_for_execution(self, _move):
        return [_move["from_position"], _move["to_position"]]

    def serialize_board(self, board):
        return list(map(lambda e: {"player": self.map_player_to_front(e[1]), "count": e[0]}, board))


    def serialize_valid_moves(self, moves, player):
        return list(map(lambda e: {
            "move": {
                "player": self.map_player_to_front(player),
                "from_position": e.from_pos,
                "to_position": e.to_pos,
                "dice": e.dice,
            },
            "next": self.serialize_valid_moves(e.next_move, player)
        }, moves))


    def roll(self, game):
        return game.get_roll()


    def return_board(self, _id, _player, _roll, _moves, _hist, _board, _multiplier, _doubling_offered, _is_finished, _winner, _suggested, _doubling_suggestion, _is_pvp):
        
        suggested = _suggested

        if (len(suggested) > 0 and type(suggested[0]) is array):
            suggested = suggested[0]

        return flask.jsonify(
            uid=_id,
            holder=self.map_player_to_front(_player),
            dice={
                "first": _roll[0],
                "second": _roll[-1]
            },
            board={
                "current_snapshot": {
                    "positions": self.serialize_board(_board)
                },
                "moves": _hist
            },
            available_moves=self.serialize_valid_moves(_moves, _player),
            suggested_moves=suggested,
            multiplier=[_multiplier[0], self.map_player_to_front(_multiplier[1])],
            doubling_offered=_doubling_offered,
            doubling_suggestion=_doubling_suggestion,
            is_finished=_is_finished,
            winner=_winner,
            is_pvp=_is_pvp,
        ).data

    