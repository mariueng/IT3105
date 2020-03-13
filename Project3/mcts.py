from operator import attrgetter


class MCTS:
    def __init__(self, node):
        self.root_node = node

    # Tree policy
    def select_node(self):
        current_node = self.root_node
        while not current_node.is_leaf_node():
            current_node = current_node.get_best_uct_child()
        return current_node

    def perform_mcts(self, no_of_simulations):
        max_eval_counter = 0
        min_eval_counter = 0
        for i in range(no_of_simulations):
            leaf_node = self.select_node()
            if leaf_node.is_terminal_node():
                evaluation = leaf_node.state.game_result()
            else:
                expanded_leaf_child_chosen_randomly = leaf_node.expand_node()
                evaluation = expanded_leaf_child_chosen_randomly.rollout()
            if evaluation > 0:
                max_eval_counter += 1
            else:
                min_eval_counter += 1
            leaf_node.backpropagate(evaluation)
        # TODO: Remove print method
        """
        print('---------------- MCTS results ------------------')
        print('Should choose maximum visited node from {}'.format(self.root_node))
        print('Simdata: number of times max won: {}, number of times min won: {}'.format(max_eval_counter,
                                                                                         min_eval_counter))
        for child in self.root_node.get_children():
            txt = "{} is visited {} times. [UCT = {}, Q(s, a) = {} , eval = {}]".format(child,
                                                                                        child.get_number_of_sims(),
                                                                                        child.uct,
                                                                                        child.get_evaluation() / child.get_number_of_sims() if child.get_number_of_sims() > 0 else 1,
                                                                                        child.get_evaluation())
            print(txt)
        print('------------------------------------------------')
        """
        return max(self.root_node.get_children(), key=attrgetter('number_of_simulations'))
