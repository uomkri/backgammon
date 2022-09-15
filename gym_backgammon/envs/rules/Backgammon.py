from copy import deepcopy
from gym_backgammon.envs.utils.move import Move


class Backgammon:
    # правило полного хода
    def get_tree_correct_depth(self, available_moves):
        if len(available_moves) == 0:
            return []

        moves_in_line = self.moves_in_line(available_moves)
        correct_depth = self.get_max_depth_of_moves(moves_in_line)

        if correct_depth > 2 or correct_depth == 1:
            return available_moves

        moves_with_correct_depth = []

        for move_1 in available_moves:
            new_move_1 = Move(move_1.from_pos, move_1.to_pos, move_1.dice, [])
            for move_2 in move_1.next_move:
                new_move_2 = Move(move_2.from_pos, move_2.to_pos, move_2.dice, [])
                new_move_1.next_move.append(new_move_2)

            moves_with_correct_depth.append(new_move_1)

        return moves_with_correct_depth

    # нужно для сетки
    def moves_in_line(self, available_moves):
        moves_in_line = []
        if len(available_moves) == 0:
            return [[]]
        # идем по первому ходу
        for i in available_moves:
            move_1 = [(i.from_pos, i.to_pos)]
            if len(i.next_move) != 0:
                for j in i.next_move:
                    move_2 = (j.from_pos, j.to_pos)
                    if len(j.next_move) != 0:
                        for z in j.next_move:
                            move_3 = (z.from_pos, z.to_pos)
                            if len(z.next_move) != 0:
                                for l in z.next_move:
                                    move_4 = (l.from_pos, l.to_pos)
                                    temp = deepcopy(move_1)
                                    move_1.append(move_2)
                                    move_1.append(move_3)
                                    move_1.append(move_4)
                                    moves_in_line.append(move_1)
                                    move_1 = temp
                            else:
                                temp = deepcopy(move_1)
                                move_1.append(move_2)
                                move_1.append(move_3)
                                moves_in_line.append(move_1)
                                move_1 = temp
                    else:
                        temp = deepcopy(move_1)
                        move_1.append(move_2)
                        moves_in_line.append(move_1)
                        move_1 = temp
            else:
                moves_in_line.append(move_1)

        moves_with_correct_depth = self.clear_moves(moves_in_line)

        return moves_with_correct_depth

    # нужно для полного хода
    def get_max_depth_of_moves(self, moves_in_line):
        max_len = 0
        for i in moves_in_line:
            if len(i) > max_len:
                max_len = len(i)

        return max_len

    def clear_moves(self, moves_in_line):
        moves_with_correct_depth = []
        max_len = self.get_max_depth_of_moves(moves_in_line)

        for i in moves_in_line:
            if len(i) >= max_len:
                moves_with_correct_depth.append(i)

        return moves_with_correct_depth
