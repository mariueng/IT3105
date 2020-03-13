import random


class TableCritic:
    def __init__(self, learning_rate_critic, decay_rate_critic, elig_discount_critic):
        self.learning_rate_critic = learning_rate_critic
        self.decay_rate_critic = decay_rate_critic
        self.elig_discount_critic = elig_discount_critic
        self.td_error = 0
        self.values = dict()
        self.ets = dict()
        self.states_in_current_episodes = []

    def create_state_value(self, state):
        self.states_in_current_episodes.append(state)
        if self.values.get(state) is None:
            self.values[state] = random.random()

    def compute_td_error(self, reinforcement, old_state, current_state):
        """
        Computes the Temporal Difference error for last state and new state
        :param reinforcement: Reinforcement factor
        :param old_state: Last state
        :param current_state: New state
        :return:
        """
        edc = self.elig_discount_critic
        vcs = self.values[current_state]
        vos = self.values[old_state]
        self.td_error = reinforcement + edc * vcs - vos
        return self.td_error

    def update_state_values(self):
        """
        Updates V(s) for all states.
        :return:
        """
        for state in self.states_in_current_episodes:
            self.values[state] = self.values[state] + self.learning_rate_critic * self.td_error * self.ets[state]

    def create_eligibility(self, state):
        """
        Initialize all eligibilities to zero.
        :param state: State to have eligibility initialised
        :return:
        """
        if self.ets.get(state) is None:
            self.ets[state] = 0

    def update_old_state_eligbility(self, old_state):
        """
        Sets the eligibility of the last explored SAP to 1.
        :param old_state: State in SAP
        :param action: Action in SAP
        :return:
        """
        self.ets[old_state] = 1

    def update_all_eligibilities(self):
        """
        Updates all eligibilities for all SAPs when a move is made.
        :return:
        """
        edc = self.elig_discount_critic
        drc = self.decay_rate_critic
        for sap in self.ets:
            self.ets[sap] = edc * drc * self.ets[sap]

    def reset_eligibilities(self):
        # Reset states in current episode as well as eligibilities
        self.states_in_current_episodes = []
        for state in self.ets:
            self.ets[state] = 0
