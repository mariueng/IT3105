import numpy as np
from operator import attrgetter
import random


class Node:
    def __init__(self, state, players_turn, c=2.5, parent=None, prev_action=None):
        self.state = state
        self.players_turn = players_turn
        self.c = c
        self.parent = parent
        self.prev_action = prev_action
        self.children = []
        self.number_of_simulations = 0
        self.eval = 0
        if self.players_turn == 1:
            uct = -1000
        else:
            uct = 1000
        self.uct = uct
        self.q_score = 0

    def get_players_turn(self):
        return self.players_turn

    def get_parent(self):
        return self.parent

    def get_prev_action(self):
        return self.prev_action

    def get_children(self):
        return self.children

    def get_number_of_sims(self):
        return self.number_of_simulations

    def get_evaluation(self):
        return self.eval

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_terminal_node(self):
        return self.state.is_game_over()

    def is_leaf_node(self):
        return len(self.children) == 0

    def get_game_state(self):
        return self.state.game_state

    def __repr__(self):
        return "(State: {} | Player {}'s turn)".format(self.get_game_state(), self.get_players_turn())

    """ MCTS methods for node """
    def get_best_uct_child(self):
        """
        Helper method for tree policy to obtain the child with the most desired uct-value,
        minimum when player 2 is choosing, maximum when player 1 is choosing.
        :param self:
        :return: node
        """
        for child in self.children:
            e_t = child.get_evaluation()
            if child.get_number_of_sims() > 0:
                # Has been simulated
                n = child.number_of_simulations
            else:
                # Child not simulated yet, i.e. we divide by zero in e_t / n. Hence, keep initiated uct.
                continue
            t = child.get_parent().get_number_of_sims()
            if child.get_players_turn() == 2:
                uct = (e_t / n) + self.c * np.sqrt(np.log(t) / (1 + n))
            else:
                uct = (e_t / n) - self.c * np.sqrt(np.log(t) / (1 + n))
            child.uct = uct
        if self.get_players_turn() == 1:
            child = max(self.get_children(), key=attrgetter('uct'))
        else:
            child = min(self.get_children(), key=attrgetter('uct'))
        return child

    def expand_node(self):
        actions = self.state.get_actions()
        for action in actions:
            next_state = self.state.move(action)
            child_node = Node(next_state, 3 - self.get_players_turn(), parent=self, prev_action=action)
            self.add_child(child_node)
        return random.choice(self.get_children())

    def rollout(self):
        current_state = self.state
        while not current_state.is_game_over():
            action_space = current_state.get_actions()
            # Should have both greedy and stochastic choice of action during rollout for best performance,
            # however, this is difficult without creating the actual nodes (to use e.g. known states for greedy choice)
            action = self.rollout_policy(action_space)
            current_state = current_state.move(action)
        return current_state.game_result()

    @staticmethod
    def rollout_policy(action_space):
        return random.choice(action_space)

    def backpropagate(self, evaluation):
        self.number_of_simulations += 1
        self.eval += evaluation
        # No need to backpropagate to parent
        if self.parent:
            self.parent.backpropagate(evaluation)
