from node import Node
import random
import numpy as np


class MCTS:
    def __init__(self, anet=None, eps=1, c=1.4, stochastic=True):
        self.root_node = None
        self.anet = anet
        self.eps = eps
        self.c = c
        self.stochastic = stochastic

    def update_root(self, new_root):
        self.root_node = new_root

    def perform_search_games(self, no_simulations):
        for g_s in range(no_simulations):
            # Use tree policy P_t to search from root to a leaf (L) of MCT. Update b_mc with each move
            leaf_node = self.tree_policy(self.root_node)
            if leaf_node.is_terminal_node():
                # Perform MCTS backpropagation from F to root.
                self.backpropagate(leaf_node, leaf_node.state.game_result())
                continue
            # Use ANET to choose rollout actions from L to a final state (F). Update b_mc with each move.
            result = self.rollout(leaf_node)
            # Perform MCTS backpropagation from F to root.
            self.backpropagate(leaf_node, result)
        distribution = self.retrieve_distribution(self.root_node)  # Should be created from b_a state
        return distribution

    def tree_policy(self, b_mc):
        while not b_mc.is_leaf_node():
            b_mc = self.get_best_child_based_on_uct(b_mc)
        if b_mc.is_terminal_node():
            return b_mc
        self.expand(b_mc)
        return self.get_best_child_based_on_uct(b_mc)

    def get_best_child_based_on_uct(self, node):
        actions = list(node.actions.keys())
        action_values = [node.get_action_values(a, self.c) for a in actions]
        return node.children[actions[np.argmax(action_values) if node.players_turn == 1 else np.argmin(action_values)]]

    def expand(self, node):
        for action in node.actions:
            child_state = node.state.get_copy()
            child_state.move(action)
            node.children[action] = Node(child_state, node, action)

    def rollout(self, leaf_node):
        current_state = leaf_node.state.get_copy()
        while not current_state.is_game_over():
            action = self.rollout_policy(current_state)
            current_state.move(action)
        return current_state.game_result()

    def rollout_policy(self, state, stochastic=True):
        action_space = state.get_actions()
        if random.random() < self.eps:
            return random.choice(action_space)
        else:
            _, stochastic_index, greedy_index = self.anet.get_distribution(state)
            if stochastic:
                return state.initial_moves[stochastic_index]
            else:
                return state.initial_moves[greedy_index]

    def backpropagate(self, leaf, result):
        current_node = leaf
        while current_node is not self.root_node:
            current_node.update_values(result)
            current_node = current_node.parent
        current_node.visits += 1

    def retrieve_distribution(self, node):
        initial_moves = node.state.initial_moves
        k = node.state.size
        D = np.zeros(k ** 2)
        for i in range(k ** 2):
            D[i] = node.actions[initial_moves[i]][0]/node.visits if initial_moves[i] in node.actions else 0
        return np.asarray(D, dtype=np.float64)
