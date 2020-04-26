from state_manager import Hex
from node import Node
from mcts import MCTS, is_terminal
from ANET import ANET
from CNN import CNN

import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt


def write(filename, cases):
    """
    Writes Replay Buffer (RBUF) to cases
    :param filename: Name of the file to be written
    :param cases: cases to write
    :return:
    """
    inputs, targets = list(zip(*cases))
    input_txt = filename + '_inputs.txt'
    target_txt = filename + '_targets.txt'
    np.savetxt(input_txt, inputs)
    np.savetxt(target_txt, targets)
    print(f'RBUF have been written to \n{input_txt}\n{target_txt}')


def load(filename):
    """
    Loads existing Replay Buffer (RBUF)
    :param filename: name of file to load
    :return: cases
    """
    inputs = np.loadtxt(filename + '_inputs.txt')
    targets = np.loadtxt(filename + '_targets.txt')
    cases = list(zip(inputs, targets))
    return cases


class Program:
    def __init__(self, episodes, simulations, initial_board, anet, mcts, number_of_saves, RBUF=None, test_data=None):
        self.episodes = episodes
        self.simulations = simulations
        self.b_a = initial_board
        self.anet = anet
        self.mcts = mcts
        self.i_s = episodes / number_of_saves
        # Training stats
        self.train_losses = []
        self.train_accuracies = []
        # Test stats
        self.test_losses = []
        self.test_accuracies = []
        self.test_data = test_data
        # Replay Buffer
        self.RBUF = RBUF if RBUF else []
        self.plt_start = 0

    def run(self, display):
        """
        Runs Deep Reinforcement Algorithm with MCTS
        :param display: True if displaying the statistics of trained neural net at the end
        :return:
        """
        eps_decay = 0.05 ** (1. / self.episodes) if self.episodes > 100 else 1
        total_start_time = time.time()
        for g_a in range(episodes):
            episode_start_time = time.time()
            # TODO: remove print
            print(f'Episode: {g_a + 1:>3}')
            if g_a % display == 0 and self.plt_start:
                self.plot(level=g_a, save=True)
            if g_a % self.i_s == 0:
                self.anet.save(size=self.b_a.size, level=g_a)
            self.b_a.reset()
            s_init = Node(self.b_a)
            self.mcts.update_root(s_init)
            while not is_terminal(self.b_a):
                D = self.mcts.perform_search_games(simulations)
                self.add_case(D)
                action = self.b_a.initial_moves[np.argmax(D)]
                self.b_a.move(self.b_a.initial_moves[np.argmax(D)])
                self.mcts.update_root(self.mcts.root_node.children[action])
            self.train_anet(g_a)
            self.mcts.eps *= eps_decay
            print(f'Episode runtime: {str(time.time() - episode_start_time)}')
        print(f'Total runtime: {str(time.time() - total_start_time)}')
        self.anet.save(size=self.b_a.size, level=g_a + 1)
        self.plot(level=g_a + 1, save=True)
        write(f'cases/size_{self.b_a.size}', self.RBUF)

    def train_anet(self, level):
        """
        Trains Neural Net (nn)
        :param level: level when nn being trained
        :return:
        """
        minibatch = math.ceil(len(self.RBUF) / 2)
        if not self.plt_start:
            self.plt_start = level
        training_cases = random.sample(self.RBUF, minibatch)
        x_train, y_train = list(zip(*training_cases))
        train_loss, train_acc = self.anet.fit(x_train, y_train)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)

    def add_case(self, D):
        """
        Adds a training case to the RBUF
        :param D: distribution
        :return:
        """
        state = self.b_a.repr_state
        self.RBUF.append((state, D))
        if random.random() > 0.5:
            self.RBUF.append(self.rotated(state, D))

    @staticmethod
    def rotated(state, D):
        """
        Rotates a training case 180 degrees to create a new training case
        :param state: state of training case
        :param D: distribution of training case
        :return: rotated array representing new case
        """
        player = state[0]
        return np.asarray([player] + list(state[:0:-1])), D[::-1]

    def plot(self, level, save=False):
        x = np.arange(len(self.train_accuracies)) + self.plt_start
        fig = plt.figure(figsize=(12, 5))
        title = f'Size: {self.b_a.size}   M: {self.simulations}   lr: {self.anet.learning_rate}   Epochs: {self.anet.epochs}   '
        title += f'RBUF size: {len(self.RBUF)}'
        fig.suptitle(title, fontsize=10)
        gs = fig.add_gridspec(1, 2)
        ax = fig.add_subplot(gs[0, 0])
        ax.set_title("Accuracy")
        ax.plot(x, self.train_accuracies, color='tab:green', label="Train")
        if self.test_data:
            ax.plot(x, self.test_accuracies, color='tab:blue', label='Test')
        plt.grid()
        plt.legend()
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title("Loss")
        ax.plot(x, self.train_losses, color='tab:red', label="Train")
        if self.test_data:
            ax.plot(x, self.test_losses, color='tab:orange', label="Test")
        plt.grid()
        plt.legend()
        if save:
            plt.savefig(f"plots/size-{self.b_a.size}")
            plt.close()
        else:
            if level == self.episodes:
                plt.show()
            else:
                plt.show(block=False)
                plt.pause(0.1)
                plt.close()


if __name__ == '__main__':
    """ Pivotal Parameters """
    # Hex
    k = 5  # Hex board size, must handle 3 <= k <= 10

    # MCTS
    episodes = 200  # I.e. number of games
    simulations = 500

    # ANET (Actor Network)
    cnn = True  # Whether to use CNN instead of ANN or not
    learning_rate = 0.001
    hidden_layers = [32, 32]  # Length is the number of layers, entry is the width of level i
    optimizer = 'Adam'  # Is currently implemented to handle Adagrad, Adam, RMSProp and SGD
    activation_function = 'ReLU'  # Is currently implemented to handle Linear, Sigmoid, Tanh, and ReLU
    epochs = 5  # Initial number of epochs to train anet on. NB! Must be greater than 0

    # Miscellaneous
    verbose = True
    m = 4  # Number of ANETs to save during run

    # Instantiate necessary objects
    initial_board = Hex(k)
    if cnn:
        anet = CNN(io_dim=k ** 2, hidden_layers=hidden_layers, learning_rate=learning_rate,
                   optimizer=optimizer, activation=activation_function, epochs=epochs)
    else:
        anet = ANET(io_dim=k ** 2, hidden_layers=hidden_layers, learning_rate=learning_rate,
                    optimizer=optimizer, activation=activation_function, epochs=epochs)
    mcts = MCTS(anet=anet)

    # Initiate program
    p = Program(episodes=episodes,
                simulations=simulations,
                initial_board=initial_board,
                anet=anet,
                mcts=mcts,
                number_of_saves=m,
                RBUF=None,
                test_data=None)

    # Run program
    p.run(display=10)

