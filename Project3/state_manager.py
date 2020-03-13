from abc import ABC, abstractmethod
import numpy as np


# State Manager Game. Controls different games based on different rules.
class Game(ABC):
    def __init__(self, game_state, turn):
        self.game_state = game_state
        self.turn = turn

    @abstractmethod
    def move(self, action):
        return Game

    @abstractmethod
    def get_actions(self):
        return list()

    @abstractmethod
    def is_game_over(self):
        return bool

    def game_result(self):
        if self.is_game_over():
            if self.turn == 2:
                return 1
            elif self.turn == 1:
                return -1
        return None

    @staticmethod
    @abstractmethod
    def print(node, player):
        return str


# To add more games to the State Manager they should be implemented here. All games must implemented a basic set of
# methods so that the game class can understand the state, produce initial game states, generated child states from a
# parent state, recognize winning states, and etc. This is ensured through inheriting the abstract class Game.
# It should not portray game specific attributes to the MCTS.

class NIM(Game):
    def __init__(self, game_state, K, turn):
        super().__init__(game_state, turn)
        self.K = K

    def move(self, action):
        next_state = np.copy(self.game_state)
        next_state -= action
        return NIM(next_state, self.K, 3 - self.turn)

    def get_actions(self):
        return list(range(1, min(self.game_state, self.K) + 1))

    def is_game_over(self):
        return self.game_state == 0

    @staticmethod
    def print(node, player):
        action = node.get_prev_action()
        left = 'Remaining stones = {:<2}'.format(node.state.game_state)
        stones = '{:<1} stones'.format(action) if action > 1 else '{:<2} stone'.format(action)
        return 'Player {} selects {:>8}: {:>21}\n'.format(player, stones, left)


class Ledge(Game):
    def __init__(self, game_state, turn):
        super().__init__(game_state, turn)

    def move(self, action):
        new_board = np.copy(self.game_state)
        if action == 0:
            assert new_board[0] != 0, 'No coin on the ledge, something went wrong!'
            new_board[0] = 0
        else:
            from_index, to_index = action
            assert new_board[from_index] != 0, 'There is no coin to move from spot {}'.format(from_index)
            assert new_board[to_index] == 0, 'There already exists a coin on spot {}'.format(to_index)

            new_board[to_index] = new_board[from_index]
            new_board[from_index] = 0

        return Ledge(new_board, 3 - self.turn)

    def get_actions(self):
        if self.game_state[0] == 2:
            return [0]
        valid_actions = []
        board = self.game_state
        board_length = len(self.game_state)
        for from_index in range(board_length - 1):
            if from_index == 0 and board[0] != 0:
                valid_actions.append(0)
                continue
            to_indexes = []
            if board[from_index + 1] != 0:
                to_index = from_index
                while to_index >= 0 and board[to_index] == 0:
                    to_indexes.append(to_index)
                    to_index -= 1
            [valid_actions.append((from_index + 1, to_index)) for to_index in to_indexes]
        return valid_actions

    def is_game_over(self):
        return list(self.game_state).count(2) == 0

    @staticmethod
    def print(node, player):
        action = node.get_prev_action()
        if action == 0:
            coin = 'copper' if node.get_parent().state.game_state[0] == 1 else 'gold'
            return 'Player {} picks up {}: {} \n'.format(player, coin, str(node.state.game_state))
        else:
            coin = 'copper' if node.get_parent().state.game_state[action[0]] == 1 else 'gold'
            return 'Player {} moves {} from cell {} to cell {}: {} \n'.format(player, coin, action[0], action[1],
                                                                              str(node.state.game_state))
