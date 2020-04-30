from node import Node
import random
import numpy as np


# Dictionary with checked terminal states. Format: {board_state : value, ...},
# where board_state is a string representation of the game and value is one of
# 0, 1, or 2 depending on the state
checked_states = dict()


def check_if_visited(state):
    """
    Helper method to check whether state has been checked for terminality before.
    :param state: State to check
    :return: 0, 1 or 2 if state is visited before, otherwise None
    """
    string_state = ''.join(str(e) for e in state.repr_state)
    visited = checked_states.get(string_state)
    if visited is not None:
        return visited
    terminal = state.is_game_over()
    if not terminal:
        checked_states[string_state] = 0
        return 0
    result = 3 - state.players_turn
    checked_states[string_state] = result
    return result


def is_terminal(state):
    """
    Checks whether a state is terminal or not
    :param state: state
    :return: True if terminal, False otherwise
    """
    check = check_if_visited(state)
    return check > 0


class MCTS:
    def __init__(self, anet=None, eps=1, c=1.4, stochastic=True):
        self.root_node = None
        self.anet = anet
        self.eps = eps
        self.c = c
        self.stochastic = stochastic

    def update_root(self, new_root):
        """
        Updates the root of the MCTS tree
        :param new_root:
        :return:
        """
        self.root_node = new_root

    def perform_search_games(self, no_simulations):
        """
        Performs search games to provide distribution for actual game
        :param no_simulations: Number of simulations to be run
        :return: Distribution over visits from root node
        """
        for g_s in range(no_simulations):
            leaf_node = self.tree_policy(self.root_node)
            if is_terminal(leaf_node.state):
                self.backpropagate(leaf_node, leaf_node.state.game_result())
                continue
            result = self.rollout(leaf_node)
            self.backpropagate(leaf_node, result)
        distribution = self.retrieve_distribution(self.root_node)  # Should be created from b_a state
        return distribution

    def tree_policy(self, b_mc):
        """
        Policy for traversing down the MCTS tree
        :param b_mc: current Monte Carlo board
        :return: Next node to expand
        """
        while not b_mc.is_leaf_node():
            b_mc = self.get_best_child_based_on_uct(b_mc)
        if b_mc.is_terminal_node():
            return b_mc
        self.expand(b_mc)
        return self.get_best_child_based_on_uct(b_mc)

    def get_best_child_based_on_uct(self, node):
        """
        Retrieves the best child of node based on UCT
        :param node: Node
        :return: Node
        """
        actions = list(node.actions.keys())
        action_values = [node.get_action_values(a, self.c) for a in actions]
        return node.children[actions[np.argmax(action_values) if node.players_turn == 1 else np.argmin(action_values)]]

    @staticmethod
    def expand(node):
        """
        Expands a node with all its potential children
        :param node: Node
        :return:
        """
        for action in node.actions:
            child_state = node.state.get_copy()
            child_state.move(action)
            node.children[action] = Node(child_state, node, action)

    def rollout(self, leaf_node):
        """
        Performs rollout from provided leaf node
        :param leaf_node: Node
        :return: Result of rollout, 1 or -1
        """
        current_state = leaf_node.state.get_copy()
        while not is_terminal(current_state):
            action = self.rollout_policy(current_state)
            current_state.move(action)
        return current_state.game_result()

    def rollout_policy(self, state, stochastic=True):
        """
        Policy for performing rollouts
        :param state: current state
        :param stochastic: True if distribution to be used is stochastic, False if greedy
        :return: Next state
        """
        action_space = state.get_actions()
        if random.random() < self.eps:
            return random.choice(action_space)
        else:
            _, stochastic_index, greedy_index = self.anet.get_distribution(state)
            if stochastic:
                return state.initial_moves[stochastic_index]
            else:
                return state.initial_moves[greedy_index]

    def backpropagate(self, leaf_node, result):
        """
        Backpropagates result up the MCTS tree
        :param leaf_node: Leaf node rollout was performed from
        :param result: Result from rollout
        :return:
        """
        current_node = leaf_node
        while current_node is not self.root_node:
            current_node.update_values(result)
            current_node = current_node.parent
        current_node.visits += 1

    def retrieve_distribution(self, node):
        """
        Retrieves the distribution for the given node, i.e. visit count of children normalized
        :param node: Node
        :return: Distribution (list)
        """
        initial_moves = node.state.initial_moves
        k = node.state.size
        D = np.zeros(k ** 2)
        for i in range(k ** 2):
            D[i] = node.actions[initial_moves[i]][0]/node.visits if initial_moves[i] in node.actions else 0
        return np.asarray(D, dtype=np.float64)
