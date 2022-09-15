from copy import deepcopy
from gym_backgammon.envs.utils.game_utils import WHITE, BLACK
from gym_backgammon.envs.utils.move import Move
from gym_backgammon.envs.utils.board import Board
from gym_backgammon.envs.rules.Backgammon import Backgammon


class KhachapuriBackgammon(Backgammon):
    def __init__(self):
        self.game_type = 'khachapuri'
        self.board = Board(self.game_type)
        self.players_positions = self.board.get_players_positions()
        self.multiplier = [1, None]
        # если 0, то игра против бота, если 1, то игра против самого себя
        self.is_pvp = False

    def get_valid_plays(self, check_player, roll, board):
        if roll is None:
            return []
        if type(roll) is int:
            roll = [roll]
        available_moves = []
        for key, (checkers, player) in enumerate(board):
            if check_player != player:
                continue

            if len(roll) > 1 and roll[0] == roll[1]:
                filtered_roll = [roll[0]]
            else:
                filtered_roll = roll

            for dice in filtered_roll:
                from_pos = key
                to_pos = key + dice

                move = self._get_move_from_position(from_pos, to_pos, check_player, self.board, dice)

                if move is not None:
                    self.execute_play(check_player, [(move.from_pos, move.to_pos)])

                    next_dices = deepcopy(list(roll))
                    next_dices.remove(dice)

                    next_moves = self.get_valid_plays(check_player, next_dices, self.board)
                    move.next_move = next_moves
                    available_moves.append(move)
                    self.restore_state(check_player, [(move.from_pos, move.to_pos)])

        available_moves_with_correct_depth = self.get_tree_correct_depth(available_moves)

        return available_moves_with_correct_depth

    def can_checker_be_offed(self, from_pos, to_pos, board, player_color):
        if to_pos == self.board.players_off_positions[player_color]:
            return True

        for i in self.board.players_home_positions[player_color]:
            if i < from_pos:
                if board[i][1] == player_color:
                    return False

        return True

    def _get_move_from_position(self, from_pos, to_pos, player, board, dice):
        # обработка для белых
        move = None
        if player == WHITE:
            # обработка выкидываний если все в доме
            if self.board.count_checkers_not_at_home(WHITE) == 0:
                if to_pos >= self.board.players_off_positions[WHITE]:
                    if self.can_checker_be_offed(from_pos, to_pos, board, WHITE):
                        move = Move(from_pos, 24, dice, [])
                else:
                    if board[to_pos][1] != BLACK:
                        move = Move(from_pos, to_pos, dice, [])
            # обработка обычного хода
            else:
                if to_pos < 24 and board[to_pos][1] != BLACK:
                    move = Move(from_pos, to_pos, dice, [])

        # обработка для черных
        else:
            to_pos = to_pos % 24
            # обработка выкидываний черных, если все в доме
            if self.board.count_checkers_not_at_home(BLACK) == 0:
                if to_pos >= self.board.players_off_positions[BLACK]:
                    if self.can_checker_be_offed(from_pos, to_pos, board, BLACK):
                        move = Move(from_pos, 24, dice, [])
                else:
                    if board[to_pos][1] != WHITE:
                        move = Move(from_pos, to_pos, dice, [])

            # обработка обычного хода
            else:
                # ходят по нижней части доски
                if to_pos > 12 and from_pos >= 12 and board[to_pos][1] != WHITE:
                    move = Move(from_pos, to_pos, dice, [])
                # ходят с нижней на верхнюю или просто по верхней части доски
                elif to_pos <= 11 and board[to_pos][1] != WHITE:
                    move = Move(from_pos, to_pos, dice, [])

        return move

    def restore_state(self, current_agent, action):
        if action:
            for move in reversed(action):
                if move:
                    from_pos = move[1]
                    to_pos = move[0]

                    # так как у нас может быть выход из 24, то проверяем на это:
                    if from_pos == 24:
                        self.board.off[current_agent] = self.board.off[current_agent] - 1
                        self.board[to_pos] = (self.board[to_pos][0] + 1, current_agent)
                    else:
                        self.board[from_pos] = (self.board[from_pos][0] - 1, current_agent)
                        if self.board[from_pos][0] < 1:
                            self.board[from_pos] = (0, None)

                        self.board[to_pos] = (self.board[to_pos][0] + 1, current_agent)

    def execute_play(self, current_agent, action):
        if action:
            # tmp нужен для assert доски
            # tmp = self.board[:]
            for move in action:
                if move:
                    from_pos = move[0]
                    to_pos = move[1]
                    # уменьшаем кол-во фишек и проверяем на 0
                    self.board[from_pos] = (self.board[from_pos][0] - 1, current_agent)
                    if self.board[from_pos][0] < 1:
                        self.board[from_pos] = (0, None)

                    # добавляем кол-во фишек и присваиваем цвет позиции
                    if to_pos == 24:
                        if current_agent == WHITE:
                            self.board.off[WHITE] += 1
                        else:
                            self.board.off[BLACK] += 1
                    else:
                        self.board[to_pos] = (self.board[to_pos][0] + 1, current_agent)

            # self.board.assert_board(action=action, board=self.board, off=self.off, game=self)