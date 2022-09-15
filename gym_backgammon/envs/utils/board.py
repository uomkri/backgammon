from gym_backgammon.envs.utils.game_utils import WHITE, NUM_POINTS, BLACK, TOKEN


class Board(list):
    def __init__(self, game_type):
        if game_type == 'khachapuri':
            board = [(0, None)] * NUM_POINTS
            board[0] = (4, WHITE)
            board[18] = (11, WHITE)
            board[12] = (4, BLACK)
            board[6] = (11, BLACK)

        elif game_type == 'long':
            board = [(0, None)] * NUM_POINTS
            board[0] = (15, WHITE)
            board[12] = (15, BLACK)

        elif game_type == 'short':
            board = [(0, None)] * NUM_POINTS
            board[0] = (2, WHITE)
            board[5] = (5, BLACK)
            board[7] = (3, BLACK)
            board[11] = (5, WHITE)
            board[12] = (5, BLACK)
            board[16] = (3, WHITE)
            board[18] = (5, WHITE)
            board[23] = (2, BLACK)
        else:
            raise Exception(f'Unsupported game type: {game_type}')


        assert 'board' in locals(), 'Wrong or non-existent game_type'

        super().__init__(board)

        self.off = [0, 0]
        self.bar = [0, 0]
        self.players_home_positions = {WHITE: [18, 19, 20, 21, 22, 23], BLACK: [6, 7, 8, 9, 10, 11]}
        self.game_type = game_type
        self.players_off_positions = {WHITE: 24, BLACK: 12}

    def get_players_positions(self):
        player_positions = [[], []]
        for key, (checkers, player) in enumerate(self):
            if player is not None and key not in player_positions:
                player_positions[player].append(key)
        return player_positions

    def make_desk_mirror(self):
        new_board = [(0, None)] * NUM_POINTS
        for key, (checkers, player) in enumerate(self):
            if player == WHITE:
                new_board[(key + 12) % 24] = tuple((checkers, BLACK))
            elif player == BLACK:
                new_board[(key + 12) % 24] = tuple((checkers, WHITE))

        temp_off = self.off[WHITE]
        self.off[WHITE] = self.off[BLACK]
        self.off[BLACK] = temp_off

        super().__init__(new_board)

    def count_checkers_not_at_home(self, agent_color):
        count = 0
        for key, (checkers, player) in enumerate(self):
            if player == agent_color and key not in self.players_home_positions[agent_color]:
                count += checkers

        return count

    def assert_board(self, action, board, off, game=None, old_board=None):
        sum_white = 0
        sum_black = 0
        for (checkers, player) in board:
            if player == WHITE:
                sum_white += checkers
            elif player == BLACK:
                sum_black += checkers


        self.print_assert(sum_white, sum_black, off, action)

        assert 0 <= sum_white <= 15 and 0 <= sum_black <= 15, self.print_assert(sum_white, sum_black, off, action)
        assert off[WHITE] < 16 and off[BLACK] < 16, self.print_assert(sum_white, sum_black, off, action)
        assert sum_white + off[WHITE] == 15, self.print_assert(sum_white, sum_black, off, action)
        assert sum_black + off[BLACK] == 15, self.print_assert(sum_white, sum_black, off, action)

    def print_assert(self, sum_white, sum_black, off, action):
        if self is not None:
            self.render()

        print(
            "sum_white={} | sum_black={} | off={} | action={}".format(sum_white, sum_black, off, action))

    def render(self):
        points = [p[0] for p in self]
        bottom_board = points[12:]
        top_board = points[:12][::-1]

        colors = [TOKEN[WHITE] if p[1] == WHITE else TOKEN[BLACK] for p in self]
        bottom_checkers_color = colors[12:]
        top_checkers_color = colors[:12][::-1]

        assert len(bottom_board) + len(top_board) == 24
        assert len(bottom_checkers_color) + len(top_checkers_color) == 24

        print("| 12 | 11 | 10 | 9  | 8  | 7  | 6  | 5  |  4 |  3 |  2 |  1 | OFF |")
        print("|--------Home Black-----------|-------P={} Outer White-------|     |".format(TOKEN[BLACK]))
        self.print_half_board(top_board, top_checkers_color, WHITE, reversed_=1)
        print("|-----------------------------|-----------------------------|     |")
        self.print_half_board(bottom_board, bottom_checkers_color, BLACK, reversed_=-1)
        print("|--------Outer Black----------|-------P={} Home White--------|     |".format(TOKEN[WHITE]))
        print("| 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | OFF |\n")

    def print_half_board(self, half_board, checkers_color, player, reversed_=-1):
        token = TOKEN[player]
        max_length = max(max(half_board), self.off[player])
        for i in range(max_length)[::reversed_]:
            row = [str(checkers_color[k]) if half_board[k] > i else " " for k in range(len(half_board))]
            off = ["{} ".format(token) if self.off[player] > i else "  "]
            row = row[:6] + row[6:] + off
            print("|  " + " |  ".join(row) + " |")

    def get_board_features(self, current_player):
        """
        - encode each point (24) with 4 units => 4 * 24 = 96
        - for each player => 96 * 2 = 192
        - 2 units indicating who is the current player
        - 2 units for white and block off checkers
        - tot = 192 + 2 + 2 = 196
        """
        features_vector = []
        for p in [WHITE, BLACK]:
            for point in range(0, NUM_POINTS):
                checkers, player = self[point]
                if player == p and checkers > 0:
                    if checkers == 1:
                        features_vector += [1.0, 0.0, 0.0, 0.0]
                    elif checkers == 2:
                        features_vector += [1.0, 1.0, 0.0, 0.0]
                    elif checkers >= 3:
                        features_vector += [1.0, 1.0, 1.0, (checkers - 3.0) / 2.0]
                else:
                    features_vector += [0.0, 0.0, 0.0, 0.0]

            features_vector += [self.off[p] / 15.0]

        if current_player == WHITE:
            # features_vector += [0.0, 1.0]
            features_vector += [1.0, 0.0]
        else:
            # features_vector += [1.0, 0.0]
            features_vector += [0.0, 1.0]
        assert len(features_vector) == 196, print("Should be 196 instead of {}".format(len(features_vector)))
        return features_vector
