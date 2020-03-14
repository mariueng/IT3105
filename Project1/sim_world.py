from board import Board


# Class for handling environment logic

# Sim worlds main job: (understand peg solitaire, little else)
# - maintaining current state of board
# - providing RL system with both initial board state and successor states of any parent state
# - Giving the reward associated with the transition from one state to another
# - determining whether a state represents a final state
# - etc.


class SimWorld:
    def __init__(self, size, shape, init_open_cells):
        self.size = size
        self.shape = shape
        self.board = Board(size, shape, init_open_cells)
        # Keep a copy to be able to reset board after each episode
        self.init_conditions = init_open_cells.copy() # inital open cells
        self.open_cells = self.board.get_open_cells()
        self.state = ''
        self.actions = dict()

    def get_state(self):
        state = ''
        for cell in self.board.get_cells_as_list():
            if cell.is_empty:
                state += '0'
            else:
                state += '1'
        return state

    def get_board(self):
        return self.board

    def get_actions(self):
        return self.actions

    def generate_successor_state(self, action):
        """
        Generates successor state of current state (board) represented by open cells, while maintaining board status.
        :param action: (from cell, jumped_cell)
        :return: to_cell, the cell jumped to
        """
        from_cell, jumped_cell = action[0], action[1]
        from_cell.remove_peg()
        jumped_cell.remove_peg()
        to_cell = self.get_to_cell(from_cell, jumped_cell)
        to_cell.add_peg()
        return to_cell

    def get_to_cell(self, from_cell, jumped_cell):
        x = from_cell.x + 2 * (jumped_cell.x - from_cell.x)
        y = from_cell.y + 2 * (jumped_cell.y - from_cell.y)
        return self.board.get_cell(x, y)

    def generate_actions(self):
        actions = {}  # Format: {from_cell: [jumped_cell, jumped_cell...] ... }
        cells = self.board.get_cells_as_list()
        for cell in cells:
            jump_nodes = []
            for neighbour in cell.get_neighbours():
                if self.check_if_valid_move(cell, neighbour):
                    jump_nodes.append(neighbour)
            actions[cell] = jump_nodes
        self.actions = actions
        return actions

    def check_if_valid_move(self, from_cell, jumped_cell):
        x = from_cell.x + 2 * (jumped_cell.x - from_cell.x)
        y = from_cell.y + 2 * (jumped_cell.y - from_cell.y)
        if (0 <= x < self.board.size) and (0 <= y < self.board.size):
            to_cell = self.board.get_cell(x, y)
            if to_cell is None:
                return False
            return not from_cell.is_empty and not jumped_cell.is_empty and to_cell.is_empty
        return False

    def get_number_of_pegs_left(self):
        return len(self.board.get_cells_as_list()) - len(self.board.get_open_cells())

    def reward(self):
        if self.get_number_of_pegs_left() == 1:
            return 500
        elif self.get_number_of_actions_left() <= 0:
            return -100
        return 0

    def get_number_of_actions_left(self):
        number_of_actions = 0
        for from_cell in self.actions.keys():
            number_of_actions += len(self.actions[from_cell])
        return number_of_actions

    def reset(self):
        self.open_cells = []
        for cell in self.board.get_cells_as_list():
            if [cell.x, cell.y] in self.init_conditions:
                self.open_cells.append(cell)
                cell.remove_peg()
            else:
                cell.add_peg()
