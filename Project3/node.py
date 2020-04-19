import numpy as np


class Node:
    def __init__(self, state, parent=None, prev_action=None):
        self.state = state
        self.players_turn = state.players_turn
        self.parent = parent
        self.visits = 0
        self.prev_action = prev_action
        # This dictionary serves as a counter for each visits an action has and the corresponding Q-score
        # Format: {action : [visits, Q-score], ...}
        self.actions = {a: [0, 0] for a in state.get_actions()}
        self.children = {}  # Format: {action : Node, ...}

    def is_terminal_node(self):
        return self.state.is_game_over()

    def is_leaf_node(self):
        return len(self.children.keys()) == 0

    def update_values(self, result):
        self.visits += 1
        # Update parent's visits
        self.parent.actions[self.prev_action][0] += 1
        n, q = self.parent.actions[self.prev_action]
        # Update parent's Q-score
        self.parent.actions[self.prev_action][1] += (result - q) / n

    def get_action_values(self, action, c):
        n, q = self.actions[action]
        c *= 1 if self.players_turn == 1 else -1
        return q + c * np.sqrt(np.log(self.visits if self.visits > 0 else 1) / (n + 1))

    def __repr__(self):
        return f"state: {self.state}"
