from state_manager import Hex
from CNN import CNN

import random
import numpy as np


class TOPP:
    def __init__(self, list_of_one_players, list_of_two_players, board_size, nr_games):
        self.list_of_one_players = list_of_one_players
        self.list_of_two_players = list_of_two_players
        self.board_size = board_size
        self.hex = Hex(board_size)
        self.standings = {player: 0 for player in self.list_of_one_players}
        self.nr_games = nr_games

    def run_tournament(self, display):
        """
        Runs Tournament of Progressive Policies (TOPP)
        :param display: If True, every game result is visualized
        :return:
        """
        print('----------- Tournament -----------')
        print(f'Size of board: {self.board_size}')
        print(f'Total number of players: {len(self.list_of_one_players)}')
        print(f'Games between each players: {self.nr_games}')
        for g in range(self.nr_games):
            for p1 in self.list_of_one_players:
                for p2 in self.list_of_two_players:
                    if p1 != p2:
                        # print('-----------    Game    -----------')
                        # print(f'{p1} - {p2} \n')
                        player_one_won, moves = self.play_topp_game(self.list_of_two_players[p1],
                                                                    self.list_of_one_players[p2], display)
                        winner = p1 if player_one_won else p2
                        self.standings[winner] += 1
                        # print(f'Agent level {winner} won after {moves}')
        print('All games finished!')
        print('The final standings are:')
        table_sorted = {player: result for player, result in sorted(self.standings.items(),
                                                                    key=lambda score: score[1],
                                                                    reverse=True)}
        index = 1
        for player in list(table_sorted.keys()):
            print(f'{index:>2}: Agent level {player:>3} - {self.standings[player]:>2} wins')
            index += 1

    def run_random_games(self, display):
        """
        Runs the provided agents against a random player
        :param display: If True, every game result is visualized
        :return:
        """
        print('----------- Matches vs Random Player -----------')
        print(f'Size of board: {self.board_size}')
        print(f'Total number of anet players: {len(self.list_of_one_players)}')
        print(f'Games between each anet and random player: {self.nr_games}')
        random_player = 'RP'
        rp_wins = 0
        for game in range(rr_games):
            for agent in self.list_of_one_players:
                player_one_won, moves = self.play_random_game(self.list_of_one_players[agent], random_player, display)
                winner = agent if player_one_won else random_player
                if winner == 'RP':
                    rp_wins += 1
                else:
                    self.standings[winner] += 1

        print('All games finished!')
        print('The final standings are:')
        print(f'Random player won {rp_wins} games')
        table_sorted = {player: result for player, result in sorted(self.standings.items(),
                                                                    key=lambda score: score[1],
                                                                    reverse=True)}
        index = 1
        for player in list(table_sorted.keys()):
            print(f'{index:>2}: Agent level {player:>3} - {self.standings[player]:>2} wins')
            index += 1

    def play_topp_game(self, player_one, player_two, display=True):
        """
        Method to play TOPP game
        :param player_one: Player 1
        :param player_two: Player 2
        :param display: If True, every game result is visualized
        :return:
        """
        self.hex.reset()
        moves = 0
        while not self.hex.is_game_over():
            moves += 1
            if self.hex.players_turn == 1:
                _, stochastic_index, greedy_index = player_one.get_distribution(self.hex)
            else:
                _, stochastic_index, greedy_index = player_two.get_distribution(self.hex)
            self.hex.move(self.hex.initial_moves[stochastic_index if random.random() > 0.5 else greedy_index])
        player_one_won = True if self.hex.game_result() == 1 else False
        if display:
            path = self.hex.is_game_over()
            self.hex.draw(path=path, animation_delay=0.2)
        return player_one_won, moves

    def play_random_game(self, anet_player, random_player, display=True):
        """
        Plays a game between agent and random player
        :param anet_player: Agent player
        :param random_player: Random player
        :param display: If True, every game result is visualized
        :return:
        """
        self.hex.reset()
        moves = 0
        while not self.hex.is_game_over():
            moves += 1
            if self.hex.players_turn == 1:
                _, stochastic_index, greedy_index = anet_player.get_distribution(self.hex)
                self.hex.move(self.hex.initial_moves[stochastic_index if random.random() > 0.5 else greedy_index])
            else:
                self.hex.move(random.choice(self.hex.get_actions()))
        anet_player_won = True if self.hex.game_result() == 1 else False
        if display:
            path = self.hex.is_game_over()
            self.hex.draw(path=path, animation_delay=0.2)
        return anet_player_won, moves


if __name__ == '__main__':
    # Hex
    k = 5  # Board size

    # Neural network(s)
    activation_functions = ["Linear", "Sigmoid", "Tanh", "ReLU"]
    optimizers = ["Adagrad", "SGD", "RMSprop", "Adam"]
    learning_rate = 0.001  # learning rate
    hidden_layers = [32, 32]  # [math.floor(2 * (1 + board_size ** 2) / 3) + board_size ** 2] * 3
    io_dim = k * k  # input and output layer sizes
    activation = activation_functions[3]
    optimizer = optimizers[3]
    epochs = 5

    # Tournament settings
    rr_games = 50
    lowest_level_agent = 0
    highest_level_agent = 200
    interval_between_agents = 50

    l = np.arange(lowest_level_agent, highest_level_agent + 1, interval_between_agents)
    models = np.sort(np.concatenate([l, l]))
    players1 = {}
    players2 = {}

    for i in range(0, len(models), 2):
        anet = CNN(io_dim, hidden_layers, learning_rate, optimizer, activation, epochs)
        anet.load(k, models[i])
        players1[models[i]] = anet
        anet = CNN(io_dim, hidden_layers, learning_rate, optimizer, activation, epochs)
        anet.load(k, models[i + 1])
        players2[models[i + 1]] = anet
    topp = TOPP(players1, players2, k, rr_games)
    # Set display=True to print every game
    topp.run_tournament(display=False)
    # Method to run ANETs against random player
    # topp.run_random_games(display=False)
