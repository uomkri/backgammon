from random import randint


WHITE = 0
BLACK = 1
NUM_POINTS = 24
OFF = 'off'
TOKEN = {WHITE: "X", BLACK: "O"}
COLORS = {WHITE: "White", BLACK: 'Black'}


def get_opponent(player):
    return BLACK if player == WHITE else WHITE


def get_winner(off):
    if off[WHITE] == 15:
        return WHITE
    elif off[BLACK] == 15:
        return BLACK
    return None


def init_board(game_type):
    if game_type == 'khachapuri':
        board = [(0, None)] * NUM_POINTS
        board[0] = (4, WHITE)
        board[18] = (11, WHITE)
        board[12] = (4, BLACK)
        board[6] = (11, BLACK)
    elif game_type == 'longgammon':
        board = [(0, None)] * NUM_POINTS
        board[0] = (15, WHITE)
        board[12] = (15, BLACK)

    assert 'board' in locals(), 'Wrong or non-existent game_type'

    return board


def get_roll():
    roll = [randint(1, 6), randint(1, 6)]
    roll = (roll[0], roll[0], roll[0], roll[0]) if roll[0] == roll[1] else roll
    return sorted(roll)


def get_all_dices_combs():
    dices_combs = set()
    dices_combs.add((6,6,6,6))
    dices_combs.add((5,5,5,5))
    dices_combs.add((4,4,4,4))
    dices_combs.add((3,3,3,3))
    dices_combs.add((2,2,2,2))
    dices_combs.add((1,1,1,1))

    for i in range(1,7,1):
        for j in range(1,7,1):
            if i == j:
                continue
            dices_combs.add(tuple(sorted([i, j])))

    return list(dices_combs)