from abc import ABC, abstractmethod
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# State Manager Game. Controls different games based on different rules.
class Game(ABC):
    def __init__(self, game_state, players_turn):
        self.game_state = game_state
        self.players_turn = players_turn

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
            return 1 if 3-self.players_turn == 1 else -1

    @staticmethod
    @abstractmethod
    def print(node, player):
        return str


# To add more games to the State Manager they should be implemented here. All games must implemented a basic set of
# methods so that the game class can understand the state, produce initial game states, generated child states from a
# parent state, recognize winning states, and etc. This is ensured through inheriting the abstract class Game.
# Any game should not portray game specific attributes to the MCTS.

class Hex(Game):
    # Neighbour dictionary for each cell: {(0, 0) : [(0, 1), (1, 0)], ...}
    neighbours = defaultdict(list)
    '''
    The ends dictionary contains the sides of the board for which the respective player is trying to win
    {Player 1: [[(0, 0), ..., (size, 0)], [(0, size), ..., (size, size)]], 
     Player 2: [[(0, 0), ..., (0, size)], [(size, 0), ..., (size, size)]]}
    '''
    ends = defaultdict(list)
    initial_moves = []
    bridge_neighbours = defaultdict(list)

    def __init__(self, size, game_state=None, players_turn=1):
        super().__init__(game_state, players_turn)
        self.size = size
        # Read sub-initialization method for more information
        if isinstance(game_state, tuple):  # Checks to create Hex game from state passed from OHT-server
            state_dict = {}
            for row in range(self.size):
                for col in range(self.size):
                    state_dict[(row, col)] = game_state[row * self.size + col]  # empty cell
            self.game_state = state_dict
        else:  # Initialize as normal Hex game
            self.game_state = game_state if game_state else self.generate_game_state()
        if len(self.neighbours) == 0:
            self.__generate_neighbours()
            self.__generate_bridge_neigbours()
        self.players_turn = players_turn

    def generate_game_state(self):
        # State is represented as dictionary: {(0, 0) : 0, (0, 1) : 2, ...}
        # NB! The action space is implicitly given at all times from the game state
        game_state = dict()
        for row in range(self.size):
            for col in range(self.size):
                game_state[(row, col)] = 0
        return game_state

    def __generate_neighbours(self):
        max_index = self.size - 1
        size = self.size
        for row in range(size):
            for col in range(size):
                self.initial_moves.append((row, col))
                if row == 0:
                    if col == 0:
                        self.neighbours[(row, col)] = [(row, col + 1), (row + 1, col)]
                    elif col == max_index:
                        self.neighbours[(row, col)] = [(row, col - 1), (row + 1, col - 1), (row + 1, col)]
                    else:
                        self.neighbours[(row, col)] = [(row, col - 1), (row, col + 1), (row + 1, col - 1), (row + 1, col)]
                elif row == max_index:
                    if col == 0:
                        self.neighbours[(row, col)] = [(row - 1, col), (row - 1, col + 1), (row, col + 1)]
                    elif col == max_index:
                        self.neighbours[(row, col)] = [(row - 1, col), (row, col - 1)]
                    else:
                        self.neighbours[(row, col)] = [(row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1)]
                else:
                    if col == 0:
                        self.neighbours[(row, col)] = [(row - 1, col), (row - 1, col + 1), (row, col + 1), (row + 1, col)]
                    elif col == max_index:
                        self.neighbours[(row, col)] = [(row - 1, col), (row, col - 1), (row + 1, col - 1), (row + 1, col)]
                    else:
                        self.neighbours[(row, col)] = [(row - 1, col), (row - 1, col + 1), (row, col - 1), (row, col + 1), (row + 1, col - 1), (row + 1, col)]
        # TODO: check this code!
        # Add left side for player 1
        self.ends[1].append([(row, 0) for row in range(size)])
        # Add right side for player 1
        self.ends[1].append([(row, size - 1) for row in range(size)])
        # Add right side for player 2
        self.ends[2].append([(0, col) for col in range(size)])
        # Add left side for player 2
        self.ends[2].append([(size - 1, col) for col in range(size)])
    
    def __generate_bridge_neigbours(self):
        """
        Generates a 2-step neighbour matrix to find bridge endpoint neighbours
        :return:
        """
        matrix = np.zeros(self.size ** 4).reshape(self.size ** 2, self.size ** 2)
        for i in range(self.size ** 2):
            for j in range(self.size ** 2):
                if i != j:
                    from_cell = (i // self.size, i % self.size)
                    to_cell = (j // self.size, j % self.size)
                    if to_cell in self.neighbours[from_cell]:
                        matrix[i][j] = 1
        two_step_matrix = np.matmul(matrix, matrix)
        np.fill_diagonal(two_step_matrix, 0)
        for i in range(self.size ** 2):
            for j in range(self.size ** 2):
                if i != j:
                    from_cell = (i // self.size, i % self.size)
                    to_cell = (j // self.size, j % self.size)
                    if two_step_matrix[i][j] == 2 and to_cell not in self.neighbours[from_cell]:
                        self.bridge_neighbours[from_cell].append(to_cell)
                    
    @property
    def repr_state(self):
        # TODO: why not a string instead of list? both hashable tho
        flat_state = [self.players_turn] + list(self.game_state.values())
        # flat_state = [self.player == 1, self.player == 2]
        # for cell in self.state.values():
        #    flat_state += [cell == 1, cell == 2]
        return np.asarray(flat_state, dtype=np.float64)

    def get_copy(self):
        return Hex(self.size, self.game_state.copy(), self.players_turn)

    def move(self, action):
        """
        Player self.turn performs the action moving the board to the next state
        :param action: (row, col)
        """
        assert self.game_state[(action[0], action[1])] == 0, "Invalid move, cell not empty"
        self.game_state[(action[0], action[1])] = self.players_turn
        self.players_turn = 3 - self.players_turn

    def get_actions(self):
        """
        Returns legal moves for the current board
        :return: list of valid actions
        """
        valid_actions = []
        for action in self.game_state.keys():
            if self.game_state[action] == 0:
                valid_actions.append(action)
        return valid_actions

    def is_game_over(self):
        """
        Checks whether a path can be found for either player
        :return: True if a path was found, False otherwise
        """
        for cell in self.ends[3 - self.players_turn][0]:
            if self.game_state[cell] == 3 - self.players_turn:
                path = self.depth_first_search(cell, [cell])
                if path:
                    return path
        return False

    def depth_first_search(self, cell, path):
        for neighbour in self.neighbours[cell]:
            if self.game_state[neighbour] == 3 - self.players_turn and neighbour not in path:
                path.append(neighbour)
                if neighbour in self.ends[3 - self.players_turn][1]:
                    return path
                else:
                    if self.depth_first_search(neighbour, path):
                        return path
        return False

    def get_minimal_path(self, path):
        i = len(path) - 1
        while True:
            if path[i] in self.ends[3 - self.players_turn][0]:
                if i != 0:
                    path = path[-1:i - 1:-1] + [path[i]]
                break
            temp_state = self.sim_copy()
            temp_state.game_state[path[i]] = 0
            if temp_state.is_game_over():
                self.game_state[path[i]] = 0
                path.remove(path[i])
            i -= 1
        return path

    def sim_copy(self):
        return Hex(self.size, self.game_state.copy(), self.players_turn)

    def cell_states(self):
        cell_states = [[], [], []]
        for cell in self.game_state:
            if self.game_state[cell] == 0:
                cell_states[0].append(cell)
            elif self.game_state[cell] == 1:
                cell_states[1].append(cell)
            elif self.game_state[cell] == 2:
                cell_states[2].append(cell)
        return cell_states

    def __cell_edges(self):
        edges = []
        for (row, col) in self.game_state:
            for (i, j) in self.game_state:
                if i == row + 1 and j == col - 1:
                    edges.append(((row, col), (i, j)))
                elif i == row + 1 and j == col:
                    edges.append(((row, col), (i, j)))
                elif i == row and j == col + 1:
                    edges.append(((row, col), (i, j)))
        return edges

    def __cell_positions(self):
        positions = {}
        for (row, col) in self.game_state:
            positions[(row, col)] = (-10 * row + 10 * col, -20 * row - 20 * col)
        return positions

    def draw(self, path=False, animation_delay=0):
        graph = nx.Graph()
        graph.add_nodes_from(self.game_state)
        graph.add_edges_from(self.__cell_edges())
        fig = plt.figure(figsize=(9, 7))
        plt.axes()
        empty, reds, blacks = self.cell_states()
        positions = self.__cell_positions()
        nx.draw(graph, pos=positions, nodelist=empty, node_color='white', edgecolors='black',
                node_size=1300 - 100 * self.size, ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=reds, node_color='red', edgecolors='black',
                node_size=1300 - 100 * self.size, ax=fig.axes[0])
        nx.draw(graph, pos=positions, nodelist=blacks, node_color='black', edgecolors='black',
                node_size=1300 - 100 * self.size, ax=fig.axes[0])
        if path:
            nx.draw(graph, pos=positions, nodelist=self.get_minimal_path(path), node_color='blue',
                    node_size=1300 - 200 * self.size, ax=fig.axes[0])
            fig.axes[0].set_title(f'Player {3 - self.players_turn} won')
        labels = {}
        i = 0
        for cell in self.game_state:
            labels[cell] = i
            i += 1
        nx.draw_networkx_labels(graph, pos=positions, labels=labels)
        if animation_delay:  # run animation automatically if delay > 0
            plt.show(block=False)
            plt.pause(animation_delay)
            plt.close()
        else:  # show single figure if delay not given
            plt.show(block=True)

    def reset(self):
        self.players_turn = 1
        for cell in self.game_state:
            self.game_state[cell] = 0

    @staticmethod
    def print(player, action):
        return f'Player {player} played {action}'

    def __repr__(self):
        return f'{self.repr_state}'
