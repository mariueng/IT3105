from state_manager import Hex
from ANET import ANET
import random
import numpy as np


class TOPP:
    def __init__(self, list_of_one_players, list_of_two_players, board_size, nr_games):
        self.list_of_one_players = list_of_one_players
        self.list_of_two_players = list_of_two_players
        self.board_size = board_size
        self.hex = Hex(board_size)
        self.table = {player: 0 for player in self.list_of_one_players}
        self.nr_games = nr_games

    def run_tournament(self, display):
        print('Tournament:')
        print(f'Board-size: {self.board_size}')
        print(f'Total number of players: {len(self.list_of_one_players)}')
        for g in range(self.nr_games):
            for p1 in self.list_of_one_players:
                for p2 in self.list_of_two_players:
                    if p1 != p2:
                        print(f'{p1} - {p2} \n')
                        player_one_won, moves = self.play_topp_game(self.list_of_two_players[p1],
                                                                    self.list_of_one_players[p2], display)
                        winner = p1 if player_one_won else p2
                        self.table[winner] += 1
                        print(f'{winner} won after {moves}')
        print('All games finished!')
        print('The final results are:')
        table_sorted = {player: result for player, result in sorted(self.table.items(),
                                                                    key=lambda score: score[1],
                                                                    reverse=True)}
        index = 1
        for player in list(table_sorted.keys()):
            print(f'{index:>2}: {player:>3} - {self.table[player]:>2} wins')
            index += 1

    def play_topp_game(self, player_one, player_two, display=True):
        self.hex.reset()
        moves = 0
        while not self.hex.is_game_over():
            _, stochastic_index, greedy_index = player_one.get_distribution(self.hex, self.hex.get_actions()) \
                if self.hex.players_turn == 1 else player_two.get_distribution(self.hex, self.hex.get_actions())
            self.hex.move(self.hex.initial_moves[stochastic_index if random.random() > 0.5 else greedy_index])
        player_one_won = True if self.hex.game_result() == 1 else False
        if display:
            path = self.hex.is_game_over()
            self.hex.draw(path=path, animation_delay=0.2)
        return player_one_won, moves


if __name__ == '__main__':
    board_size = 5

    activation_functions = ["linear", "sigmoid", "tanh", "relu"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    alpha = 0.001  # learning rate
    H_dims = [128, 128, 64, 64]  # [math.floor(2 * (1 + board_size ** 2) / 3) + board_size ** 2] * 3
    io_dim = board_size * board_size  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 10

    num_games = 100
    bottom_level = 0
    top_level = 200
    interval = 50

    l = np.arange(bottom_level, top_level + 1, interval)
    models = np.sort(np.concatenate([l, l]))
    players1 = {}
    players2 = {}

    for i in range(0, len(models), 2):
        ann = ANET(io_dim, H_dims, alpha, optimizer, activation, epochs)
        ann.load(board_size, models[i])
        players1[models[i]] = ann
        ann = ANET(io_dim, H_dims, alpha, optimizer, activation, epochs)
        ann.load(board_size, models[i + 1])
        players2[models[i + 1]] = ann
    tournament = TOPP(players1, players2, board_size, num_games)
    tournament.run_tournament(display=False)