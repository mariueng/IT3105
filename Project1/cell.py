class Cell:
    """
    Creates a cell representing a single cell on the board
    """
    def __init__(self, x, y, s, is_open_cell):
        """
        Initializes cell with position
        :param x: row position
        :param y: column position
        """
        self.x = x
        self.y = y
        self.is_empty = False
        self.is_open_cell = is_open_cell
        if is_open_cell:
            self.is_empty = True  # Terminal cells starts out being empty
        self.neighbours = []
        # Only for testing purposes
        self.name = s

    # TODO: remove testing properties
    # Method for obtaining name during testing
    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_neighbours(self):
        return self.neighbours

    def add_neighbours(self, neighbours, grid, size):
        """
        Method for adding neighbours to cell
        :param neighbours: Neighbours to be added
        :param grid: Grid of board which the cell is present in
        :param size: Size of current board/grid which the cell is present in
        :return:
        """
        for cell in neighbours:
            if 0 <= self.x + cell[0] < size and 0 <= self.y + cell[1] < size:
                neighbour = grid[self.y + cell[1], self.x + cell[0]]
                if not self.is_neighbour(neighbour):
                    if neighbour is not None:
                        self.neighbours.append(neighbour)

    def is_neighbour(self, cell):
        return self.neighbours.__contains__(cell)

    def remove_peg(self):
        self.is_empty = True

    def add_peg(self):
        self.is_empty = False

    def cell_is_empty(self):
        return self.is_empty

    def __repr__(self):
        return str(self.name)
