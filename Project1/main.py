from visualize import Visualize
from simWorldPeg import SimWorld
from actor import Actor
from table_critic import TableCritic
from nn_critic import NNCritic
import time
import matplotlib.pyplot as plt


def run_peg(shape_: str, size_: int, open_cells_: list, episodes_: int, learner_: str, hidden_layers_: list,
            l_rates: list, ed_rates_: list, d_factors: list, epsilon_: float, eps_decay_: float, display_: bool,
            delay_: float):
    """
    Critic-Actor Algorithm for solving Peg Solitaire.
    :param shape_: Shape of board, triangle ('t') or diamond ('d')
    :param size_: Size of board
    :param open_cells_: The open cell(s) in the initial state of the puzzle
    :param episodes_: Number of episodes to run during training
    :param learner_: Type of learner (critic)
    :param hidden_layers_: Dimensions of the net (excluding the input layer)
    :param l_rates: Learning rates for actor and critic, respectively
    :param ed_rates_: Eligibility decay rate for actor and critic, respectively
    :param d_factors: Discount factors for actor and critic, respectively
    :param epsilon_: Initial value of epsilon
    :param eps_decay_: Epsilon decay rate
    :param display_: Display variable, true if displaying training
    :param delay_: Delay between frames of the visualization
    :return:
    """
    # Initialize SimWorld, this creates a board, and retrieves current state and action space.
    sw = SimWorld(size_, shape_, open_cells_)
    pegs_left = []
    runs = []
    if learner_ == 'table':
        critic = TableCritic(l_rates[1], ed_rates_[1], d_factors[1])
    else:
        state = sw.get_state()
        input_layer_size = len(state)
        critic = NNCritic(l_rates[1], ed_rates_[1], d_factors[1], input_dim=input_layer_size,
                          hidden_layers=hidden_layers_)
    actor = Actor(l_rates[0], ed_rates_[0], d_factors[0])
    # Create Visualize class to keep track of visualization of board.
    vis = Visualize(sw.get_board(), delay_)
    # Initialize critic with V(s) = small random values.
    # critic.init_state_value(current_state)
    # Initialize actor with Π(s, a) = 0 for SAPs.
    # actor.init_saps(current_state, action_space)
    start_time = time.time()
    # Repeat for each episode
    for episode in range(1, episodes_ + 1):
        # Reduce epsilon after halfway through to shift from explore to exploit
        if episode >= episodes_ / 2:
            epsilon_ = epsilon_ * eps_decay_
        print('Episode: ' + str(episode))
        # Reset board to initial state
        sw.reset()
        # Reset eligibilities in both critic and actor
        critic.reset_eligibilities()
        actor.reset_eligibilities()
        # Initialize
        action_space = sw.generate_actions()
        current_state = sw.get_state()
        # Create state values and eligibilities for critic
        if learner_.lower() == 'table':
            critic.create_state_value(current_state)
            critic.create_eligibility(current_state)

        # Create SAPs and elgibilities for actor
        actor.create_saps(current_state, action_space)
        actor.create_eligibilities(current_state, action_space)
        # Choose initial action
        action = actor.get_action(current_state, action_space, epsilon_)
        # The episode is over when there are no more actions available.
        while len(action_space) > 0:
            # Repeat for each step in the episode
            old_state = current_state
            # Perform action
            to_cell = sw.generate_successor_state(action)
            current_state = sw.get_state()
            action_space = sw.generate_actions()
            if learner_.lower() == 'table':
                critic.create_state_value(current_state)
                critic.create_eligibility(current_state)
            actor.create_eligibilities(current_state, action_space)
            actor.create_saps(current_state, action_space)
            reinforcement = sw.reward()
            """ Uncomment line below to visualize during training """
            # vis.update_graph(to_cell, action)
            # Actor: a' <- Π(s') the action dictated by the current policy for state s'
            next_action = actor.get_action(current_state, action_space, epsilon_)
            # Actor: set eligibility for old_state
            actor.update_last_explored_et(old_state, action)
            # Critic: Compute TD Error
            td_error = critic.compute_td_error(reinforcement, old_state, current_state)
            # For all SAPs in the current episode do (PSEUDO-CODE line 6(a) - (d))
            if learner_ == 'table':
                # Critic: Update eligibility for old_state
                critic.update_old_state_eligbility(old_state)
                # Critic update V(s) and e(s)
                critic.update_state_values()
            else:
                critic.fit(reinforcement, old_state, current_state, td_error)
            # Done regardless of type of learner
            critic.update_all_eligibilities()
            # Actor update Π(s, a) and e(s, a)
            actor.update_saps(td_error)
            actor.update_all_eligibilities()
            action = next_action
            if action is None:
                break
        # Store pegs left after each episode with episode number
        pegs_left.append(sw.get_number_of_pegs_left())
        runs.append(episode)

    time_spent = time.time() - start_time
    print('Total run time during training: ' + str(time_spent))
    plt.plot(runs, pegs_left)
    plt.show()

    """ Run greedy algorithm (epsilon = 0) """
    start_time = time.time()
    sw.reset()
    # If not visualize object is instantiated before testing, uncomment line below
    # vis = Visualize(sw.get_board(), delay)
    to_cell = None
    state = sw.get_state()
    valid_actions = sw.generate_actions()
    action = actor.get_action(state, valid_actions, epsilon=0)
    while len(valid_actions) > 0:
        to_cell = sw.generate_successor_state(action)
        state = sw.get_state()
        vis.update_graph(to_cell, action)
        valid_actions = sw.generate_actions()
        actor.create_saps(state, valid_actions)
        action = actor.get_action(state, valid_actions, epsilon=0)
        if action is None:
            break

    time_spent = time.time() - start_time
    print('Total run time during testing: ' + str(time_spent))


if __name__ == '__main__':
    shape = 't'
    size = 5
    open_cells = [[1, 2]]
    episodes = 500
    # Set to 'table' for table lookup or 'nn' for neural network critic.
    learner = 'table'
    # Should only contain sizes of hidden layers. Input layers is found by the program given diamond or triangle shape.
    # Format: [hidden layer 1, hidden layer 2, ...]
    hidden_layers = [5, 1]
    learning_rates = [0.1, 0.1]
    ed_rates = [0.9, 0.9]
    disc_factors = [0.95, 0.95]
    epsilon = 0.05
    # To reduce epsilon after many runs
    eps_decay = 0.95
    display = False
    delay = 0.2
    run_peg(shape, size, open_cells, episodes, learner, hidden_layers, learning_rates, ed_rates, disc_factors, epsilon,
            eps_decay, display, delay)
