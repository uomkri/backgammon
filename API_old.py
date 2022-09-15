import logging
import random
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import uuid
import pymongo


app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'

player = 0

client = pymongo.MongoClient(
    "mongodb+srv://user0:NiggaNigga25@backgammon.d9g5e.mongodb.net/backgammon?retryWrites=true&w=majority")
db = client.backgammon

def put_new_session():
    uid = str(uuid.uuid4())
    game = backgammon.Backgammon()
    _board = game.board
    _off = game.off
    _curr_player = 0

    entry = {
        "uid": uid,
        "board": _board,
        "off": _off,
        "curr_player": _curr_player,
        "history": []
    }
    res = db.games.insert_one(entry)
    print(res)
    return [uid, game]


def get_session(uid):
    res = db.games.find_one({"uid": uid})
    print(res)
    board = res["board"]
    off = res["off"]
    game = backgammon.Backgammon()
    game.board = board
    game.off = off
    return [res["curr_player"], game, res["history"]]


def update_game(uid: str, board, off, curr_player, moves):
    print(list(moves))
    res = db.games.update_one({"uid": uid}, {
        "$set": {
            "board": board,
            "off": off,
            "curr_player": curr_player,
            "history": moves
        }
    })
    print(res)


def map_player(p):
    if p == 0:
        return 2
    elif p == 1:
        return p


def serialize_board(board):
    return list(map(lambda e: {"player": map_player(e[1]), "count": e[0]}, board))


def serialize_valid_moves(moves):
    return list(map(lambda e: {
        "move": {
            "player": map_player(player),
            "from_position": e.from_pos,
            "to_position": e.to_pos,
            "dice": e.dice,
        },
        "next": serialize_valid_moves(e.next_move)
    }, moves))


def roll(game):
    return game.get_roll()


def move_agent(uid: str):
    res = get_session(uid)
    game = res[1]
    _player = res[0]
    history = res[2]
    _board = game.board
    _roll = roll(game)
    _moves = game.get_valid_plays(_player, _roll, _board)
    in_line = game.moves_in_line(_moves)

    index = random.randint(0, len(in_line) - 1)
    selected_moves = in_line[index]

    game.render()
    print("BEFORE")

    rec = []
    for e in selected_moves:
        rec.append({
            "player": map_player(_player),
            "from_position": e[0],
            "to_position": e[1],
            "dice": 1
        })
    print("SELECTED MOVES")
    print(selected_moves)

    game.execute_play(_player, selected_moves)

    for e in rec:
        history.append(e)
    new_board = game.board
    next_player = 0
    new_off = game.off

    print(new_board)
    game.render()

    update_game(
        uid=uid,
        board=new_board,
        curr_player=next_player,
        moves=history,
        off=new_off
    )


@app.route("/start/<name>")
@cross_origin()
def start(name):
    if name == "khachapuri":
        # session_id = state.add_session()
        res = put_new_session()
        uid = res[0]
        game = res[1]

        current_roll = roll(game)
        board = game.board
        valid_moves = game.get_valid_plays(
            check_player=0,
            roll=current_roll,
            board=board
        )
        print("START")
        res = return_board(
            _id=uid,
            _player=0,
            _roll=current_roll,
            _moves=valid_moves,
            _hist=[],
            _board=board,
            _is_finished=False
        )
        return res
    else:
        return jsonify(
            message="Unknown type"
        ).headers.add('Access-Control-Allow-Origin', "*")


@app.route("/move/<uid>", methods=['POST'])
@cross_origin()
def move(uid):
    print(uid)

    _uid = str(uid)

    req = request.get_json()

    mapped = map(lambda e: map_move_for_execution(e), req)

    rec = []
    for e in req:
        rec.append({
            "player": map_player(player),
            "from_position": e["from_position"],
            "to_position": e["to_position"],
            "dice": 1
        })

    game_res = get_session(uid)
    _player = game_res[0]
    game = game_res[1]
    history = game_res[2]

    game.execute_play(player, mapped)

    for item in rec:
        history.append(item)

    update_game(
        uid=uid,
        board=game.board,
        curr_player=1,
        off=game.off,
        moves=history
    )

    move_agent(_uid)

    game_res = get_session(uid)
    _player = game_res[0]
    game = game_res[1]
    history = game_res[2]

    _roll = roll(game)

    moves = game.get_valid_plays(player, _roll, game.board)

    winner = game.get_winner()
    is_finished = False
    if winner is not None:
        print('У НАС ЕСТЬ ПОБЕДИТЕЛЬ!!!', winner)
        is_finished = True
        game.render()

    try:
        res = return_board(
            _id=_uid,
            _player=map_player(player),
            _roll=_roll,
            _moves=moves,
            _hist=history,
            _board=game.board,
            _is_finished=is_finished
        )  # .headers.add('Access-Control-Allow-Origin', "*")
        return res
    except Exception as e:
        logging.exception(e)
        return ""


def map_move_for_execution(_move):
    return [_move["from_position"], _move["to_position"]]


def return_board(_id, _player, _roll, _moves, _hist, _board, _is_finished):
    return flask.jsonify(
        uid=_id,
        holder=_player,
        dice={
            "first": _roll[0],
            "second": _roll[-1]
        },
        board={
            "current_snapshot": {
                "positions": serialize_board(_board)
            },
            "moves": _hist
        },
        available_moves=serialize_valid_moves(_moves),
        is_finished=_is_finished
    ).data
