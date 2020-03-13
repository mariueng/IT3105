from state_manager import NIM
from state_manager import Ledge
from node import Node
from mcts import MCTS
import random
import time



def set_first_player(player):
    assert player == 1 or player == 2 or player == 3, "Please provide a valid first player option (1, 2 or 3)"
    if player == 3:
        return random.choice([1, 2])
    return player


def perform_action(no_of_simulations, game_state):
    mcts = MCTS(game_state)
    return mcts.perform_mcts(no_of_simulations)


def run(G, M, N, K, B, P, game_type, _verbose):
    win_stats = {1: 0, 2: 0}
    start_stats = {1: 0, 2: 0}
    verbose_text = ""
    current_state = None
    first_player = 0
    # For each episode/game, do
    start_time = time.time()
    for i in range(1, G + 1):
        print('**************** GAME NUMBER {} ****************'.format(i))
        first_player = set_first_player(P)
        start_stats[first_player] += 1
        """ Add access to more games here """
        if game_type == 0:
            current_state = Node(NIM(N, K, first_player), first_player)
        elif game_type == 1:
            current_state = Node(Ledge(B, first_player), first_player)
        verbose_text += "Initial state: {} \n".format(current_state.get_game_state())
        # create new game to play
        move_index = 0
        player = 3 - first_player
        while not current_state.is_terminal_node():
            player = 3 - player
            # Perform action based on MCTS
            current_state = perform_action(M, current_state)
            move_index += 1
            if _verbose:
                verbose_text += str(move_index) + ": "
                """ Add verbose for more games here """
                if game_type == 0:
                    verbose_text += NIM.print(current_state, player)
                else:
                    verbose_text += Ledge.print(current_state, player)
        verbose_text += "Player " + str(player) + " won \n\n"
        win_stats[player] += 1
    if _verbose:
        print(verbose_text)
        print('Run time: ' + str(time.time() - start_time))
    if P == 3:
        print("Player {} started {} and won {}/{}, while player {} started {} and won {}/{} games."
              .format(1, start_stats[1], win_stats[1], G, 2, start_stats[2], win_stats[2], G))
    else:
        print("Player {} started and won {}/{} ({:.4} %) games"
              .format(first_player, win_stats[first_player], G, 100 * win_stats[first_player] / G))


if __name__ == '__main__':
    G = 50
    M = 600
    # Need to be able to handle 100 > N > K > 1
    N = 12
    K = 3
    # Board for Ledge, need to be able to handle 1 <= length <= 20
    # and 0 <= number of copper coins <= L - 1
    # B = [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 2, 1, 0, 1]
    B = [0, 1, 0, 2, 0, 0, 1]
    # P = 1: Player 1 starts, P = 2: Player 2 starts, P = 3: random
    P = 1
    # When adding more games just increment the index for the next game
    type_of_game = 0  # [0: NIM, 1: Ledge]
    verbose = True
    run(G, M, N, K, B, P, type_of_game, verbose)
