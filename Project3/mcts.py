from node import Node
import random
import numpy as np


class MCTS:
    def __init__(self, root_node, eps=1):
        self.root_node = root_node
        self.eps = eps

    def update_root(self, new_root):
        self.root_node = new_root

    def perform_search_games(self, no_simulations, anet):
        for g_s in range(no_simulations):
            # Initialize Monte Carlo game board (b_mc) to same state as root
            # NB! Has to be a copy!
            # b_mc = Node(self.root_node.state.get_copy())
            # Use tree policy P_t to search from root to a leaf (L) of MCT. Update b_mc with each move
            leaf_node = self.tree_policy(self.root_node)
            if leaf_node.is_terminal_node():
                # Perform MCTS backpropagation from F to root.
                self.backpropagate(leaf_node, leaf_node.state.game_result())
                continue
            # Use ANET to choose rollout actions from L to a final state (F). Update b_mc with each move.
            result = self.rollout(leaf_node, anet)
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

    @staticmethod
    def get_best_child_based_on_uct(node, c=1.4):
        actions = list(node.actions.keys())
        action_values = [node.get_action_values(a, c) for a in actions]
        return node.children[actions[np.argmax(action_values) if node.players_turn == 1 else np.argmin(action_values)]]

    def expand(self, node):
        for action in node.actions:
            child_state = node.state.get_copy()
            child_state.move(action)
            node.children[action] = Node(child_state, node, action)

    def rollout(self, leaf_node, anet):
        current_state = leaf_node.state.get_copy()
        while not current_state.is_game_over():
            action = self.rollout_policy(current_state, anet, eps=self.eps)
            current_state.move(action)
        return current_state.game_result()

    def rollout_policy(self, state, anet, eps=1, stochastic=True):
        action_space = state.get_actions()
        if random.random() < eps:
            return random.choice(action_space)
        else:
            _, stochastic_index, greedy_index = anet.get_distribution(state, action_space, state.initial_moves)
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
