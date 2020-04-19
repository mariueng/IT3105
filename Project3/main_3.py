from state_manager import Hex
from node import Node
from mcts import MCTS
from ANET import ANET
import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt


# Global variables to check losses and accuracies of model
all_cases = []

test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []


def train_anet(net, buffer, minibatch_size_):
    """
    Trains Neural Net (nn)
    :param net: Neural network to be trained
    :param buffer: Replay buffer storing training cases
    :param minibatch_size_: Size of minibatch to train nn on
    :return:
    """
    minibatch_size_ = math.ceil(len(all_cases) / 2)
    training_cases = random.sample(all_cases, minibatch_size_)
    x_train, y_train = list(zip(*training_cases))
    x_test, y_test = list(zip(*all_cases))
    train_loss, train_acc = net.fit(x_train, y_train)
    test_loss, test_acc = net.get_status(x_test, y_test)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)


def write(filename, cases):
    inputs, targets = list(zip(*cases))
    input_txt = filename + '_inputs.txt'
    target_txt = filename + '_targets.txt'
    np.savetxt(input_txt, inputs)
    np.savetxt(target_txt, targets)
    print(f'Buffer have been written to \n{input_txt}\n{target_txt}')


def load(filename):
    inputs = np.loadtxt(filename + '_inputs.txt')
    targets = np.loadtxt(filename + '_targets.txt')
    cases = list(zip(inputs, targets))
    return cases


def plot(episode, save=False):
    episodes_ = np.arange(episode)
    fig = plt.figure(figsize=(12, 5))
    title = f'Size: {k}   M: {m}   lr: {learning_rate}   Epochs: {epochs}   '
    title += f'Batch size: {minibatch_size}   All cases: {len(all_cases)}'
    fig.suptitle(title, fontsize=10)
    gs = fig.add_gridspec(1, 2)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("Accuracy")
    ax.plot(episodes_, train_accuracies, color='tab:green', label="Batch")
    ax.plot(episodes_, test_accuracies, color='tab:blue', label="All cases")
    plt.legend()
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("Loss")
    ax.plot(episodes_, train_losses, color='tab:red', label="Batch")
    ax.plot(episodes_, test_losses, color='tab:orange', label="All cases")
    plt.legend()
    if save:
        plt.savefig(f"plots/size-{k}")
        plt.close()
    else:
        if episode == episodes:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(0.1)
            plt.close()


def plot_level_accuracies(levels, anet):
    cases = load(f"cases/size_{k}")
    losses = []
    accuracies = []
    for l in levels:
        anet.load(k, l)
        input_, target = list(zip(*cases))
        losses.append(anet.get_loss(input_, target))
        accuracies.append(anet.accuracy(input_, target))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel("episodes")
    fig.axes[0].set_title(f"Size {k}")
    ax.plot(levels, accuracies, color='tab:blue', label="Accuracy")
    ax.plot(levels, losses, color='tab:orange', label="Loss")
    plt.legend()
    plt.show()


def add_case(b_a, D):
    state = b_a.repr_state
    size = b_a.size
    all_cases.append((state, D))
    if random.random() > 0.5:
        player = state[0]
        state = state[1:].reshape(size, size)
        rot_state = np.rot90(state, k=2, axes=(0, 1))
        probs = D.reshape(size, size)
        rot_D = np.rot90(probs, k=2, axes=(0, 1))
        all_cases.append((np.asarray([player] + list(rot_state.reshape(size ** 2))), rot_D.reshape(size ** 2)))


def win_rate(p1, p2):
    game = Hex(k)
    wins = np.zeros(100)
    for i in range(100):
        p1_starts = bool(i % 2)
        game.reset()
        move = p1.get_greedy(game) if p1_starts else p2.get_greedy(game)
        game.move(move)
        turn = not p1_starts
        while not game.is_game_over():
            move = p1.get_greedy(game) if turn else p2.get_greedy(game)
            game.move(move)
            turn = not turn
        if (p1_starts and game.result() == 1) or (not p1_starts and game.result() == -1):
            wins[i] = 1
    return sum(wins)/100


def run(episodes_, simulations_, k_, first_player_, learning_rate_, hidden_layers_, activation_function_, optimizer_,
        epochs_, m_, rr_games_, minibatch_size_, verbose_, live_plot=False):
    # Save interval and training cases stored in these lists
    i_s = episodes_ / m_
    RBUF = []

    # Epsilon decay options
    if episodes_ >= 200:
        eps_decay = 0.05 ** (1./episodes_)
    else:
        eps_decay = 0.99

    # Initialize ANET
    anet = ANET(k_ * k_, hidden_layers_, learning_rate_, optimizer_, activation_function_, epochs_)
    anet.save(k_, 0)

    # Stats for plotting and verbose
    losses = []
    accuracies = []
    win_stats = {1: 0, 2: 0}
    start_stats = {1: 0, 2: 0}
    verbose_text = ""

    # For each episode/game, do
    start_time = time.time()
    for g_a in range(episodes_):
        print(f'**************** GAME NUMBER {g_a} ****************')
        # Live plotting
        if live_plot:
            plot(episode=g_a)
        if g_a % i_s == 0:
            anet.save(k_, level=g_a)
            anet.epochs += 10
            plot(episode=g_a, save=True)
        # Initialize the actual game board (b_a) to an empty board.
        b_a = Hex(size=k_, game_state=None, players_turn=first_player_)
        # TODO: check if reset of board is necessary: b_a.reset()
        # s_init ← starting board state
        s_init = Node(b_a)
        # Initialize the Monte Carlo Tree (MCT) to a single root, which represents s_init
        mcts = MCTS(s_init)
        while not b_a.is_game_over():
            # D = distribution if visit counts in MCT along all arcs emanating from root
            D = mcts.perform_search_games(simulations_, anet)
            # Add case (root, D) to RBUF
            add_case(b_a, D)
            # Choose actual move (a*) based on D
            action = b_a.initial_moves[np.argmax(D)]  # Distribution is corrected for possible action at current b_a
            # Perform a* on root to produce successor state s*
            successor_state = mcts.root_node.children[action]
            # Update b_a to s*
            b_a.move(action)
            # In MCT, retain subtree rooted at s*; discard everthin else
            # root <- s*
            mcts.update_root(successor_state)
        # Train ANET on a minibatch of case in RBUF
        train_anet(anet, RBUF, minibatch_size_)
        mcts.eps *= eps_decay
    anet.save(size=k_, level=g_a + 1)
    plot(episode=g_a + 1, save=True)
    write(f'cases/size_{k_}', all_cases)
    if verbose_:
        print(verbose_text)
        print('Run time: ' + str(time.time() - start_time))


if __name__ == '__main__':
    """ Pivotal Parameters """
    # MCTS
    episodes = 50  # I.e. number of games
    simulations = 500

    # Hex
    k = 5  # Hex board size, must handle 3 <= k <= 10
    first_player = 1  # p = 1: Player 1 starts, p = 2: Player 2 starts, p = 3: random

    # ANET
    learning_rate = 0.005
    hidden_layers = [120, 84]  # Length is the number of layers, entry is the width of level i
    optimizer = 'Adam'  # Is currently implemented to handle Adagrad, Adam, RMSProp and Adagrad
    activation_function = 'relu'  # Could be linear, sigmoid, tanh, relu, etc..
    epochs = 0

    # TOPP
    m = 4
    rr_games = 25
    minibatch_size = 6

    verbose = True
    run(episodes, simulations, k, first_player, learning_rate, hidden_layers, activation_function, optimizer, epochs,
        m, rr_games, minibatch_size, verbose)
