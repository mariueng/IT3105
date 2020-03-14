import random


class Actor:
    def __init__(self, learning_rate_actor, decay_rate_actor, elig_discount_actor):
        """
        Initializes Actor with the given parameters.
        :param learning_rate_actor: Learning rate for actor (i.e, alpha)
        :param decay_rate_actor: Trace decay for eligibilities (i.e., lambda), between [0, 1]
        :param elig_discount_actor: Discount factor (i.e., gamma), between [0, 1]
        """
        self.learning_rate_actor = learning_rate_actor
        self.trace_decay_actor = decay_rate_actor
        self.elig_discount_actor = elig_discount_actor
        # Dictionary containing State-Action-Pairs (SAPs). Format: {(State, Action): z, ...} where z is a real number.
        self.saps = dict()
        # Dictionary containing Eligibility traces for all SAPs. Format: {(State, Action): e, ...} where e is a number
        # between 0 and 1.
        self.ets = dict()
        # Keep track of SAPs in current episode
        self.saps_in_current_episode = []

    def create_saps(self, state, actions):
        """
        Initializes all State-Action-Pairs (SAPs) with a real number (zero).
        :param state: current board state
        :param actions: List
        :return:
        """
        for from_cell in actions:
            for to_cell in actions[from_cell]:
                sap = (state, (from_cell, to_cell))
                self.saps_in_current_episode.append(sap)
                if self.saps.get(sap) is None:
                    self.saps[sap] = 0

    def update_saps(self, td_error):
        """
        Updates all SAPs value, updating and maintaining the policy.
        :return:
        """
        lra = self.learning_rate_actor
        for sap in self.saps_in_current_episode:
            s = self.saps[sap]
            et = self.ets[sap]
            self.saps[sap] = s + lra * td_error * et

    def create_eligibilities(self, state, actions):
        """
        Initialize all eligibilities to zero.
        :param state: State in SAP to have eligibility initialised
        :param actions: Action in SAP to have eligibility initialised
        :return:
        """
        for from_cell in actions:
            for to_cell in actions[from_cell]:
                if self.ets.get((state, (from_cell, to_cell))) is None:
                    self.ets[(state, (from_cell, to_cell))] = 0

    def update_last_explored_et(self, state, action):
        """
        Sets the eligibility of the last explored SAP to 1.
        :param state: State in SAP
        :param action: Action in SAP
        :return:
        """
        self.ets[(state, action)] = 1

    def update_all_eligibilities(self):
        """
        Updates all eligibilities for all SAPs when a move is made.
        :return:
        """
        for sap in self.ets:
            self.ets[sap] = self.elig_discount_actor * self.trace_decay_actor * self.ets[sap]

    def reset_eligibilities(self):
        # Reset SAPs in current episode as well
        self.saps_in_current_episode = []
        for sap in self.saps:
            self.ets[sap] = 0

    # Finds the "best" action based on SAPs (Q) that have been corrected with eligibility traces and TD error.
    def get_action(self, state, valid_actions, epsilon):
        actions = {}
        for from_cell in valid_actions:
            for to_cell in valid_actions[from_cell]:
                actions[(from_cell, to_cell)] = self.saps[(state, (from_cell, to_cell))]
        if len(actions) > 0:
            if random.random() < epsilon:
                return random.choice(list(actions.items()))[0]
            else:
                return max(actions, key=actions.get)
