import numpy as np
from Project1.cell import Cell


class Board:
    """
    Creates a board for the game given parameters as size and shape.
    """
    def __init__(self, size: int, shape: str, init_open_cells: list):
        """
        Set parameters for shape and size
        :param size: Size of the board (i.e., including all numbers in [3, 10])
        :param shape: Shape of the board (diamond or triangle)
        """
        self.size = size
        self.type = shape
        self.grid = np.ndarray(shape=(size, size), dtype=Cell)
        self.init_open_cells = init_open_cells  # Format: [[x, y], ...]
        self.init_board()

    def init_board(self):
        """
        Initialize board depending on provided shape
        :return:
        """
        if self.type.lower() == 'd':
            self.init_diamond_board()
        elif self.type.lower() == 't':
            self.init_triangle_board()
        else:
            print("Invalid board, enter 'd' for diamond or 't' for triangular board")

    def init_diamond_board(self):
        # For testing purposes only
        s = 1
        """
        Initializes the board given that the shape is a diamond
        :return:
        """
        for r in range(self.size):
            for c in range(self.size):
                is_open = False
                if [c, r] in self.init_open_cells:
                    is_open = True
                cell = Cell(c, r, s, is_open)
                self.grid[r, c] = cell
                s += 1

        # List of potential neighbours for each cell, board shape restrictions are handled in the cell class.
        # After a 45° CCW rotation the neighbours are represented as [W, SW, S, N, NE, E].
        neighbours = [[-1, 0], [-1, 1], [0, 1], [0, -1], [1, -1], [1, 0]]
        for r in range(self.size):
            for c in range(self.size):
                self.grid[r, c].add_neighbours(neighbours, self.grid, self.size)

    def init_triangle_board(self):
        # For testing purposes only
        s = 1
        """
        Initializes the board given that the shape is a triangle.
        :return:
        """
        for r in range(self.size):
            for c in range(r + 1):
                is_open = False
                # Check if this cell is in list of open cells
                if [c, r] in self.init_open_cells:
                    is_open = True
                cell = Cell(c, r, s, is_open)
                self.grid[r, c] = cell
                s += 1
        # List of potential neighbours for each cell, board shape restrictions are handled in the cell class.
        # After a transformation from equilateral triangle to array the neighbours are represented as
        # [NW, W, N, S, E, SE].
        neighbours = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 1]]
        for r in range(self.size):
            for c in range(r + 1):
                self.grid[r, c].add_neighbours(neighbours, self.grid, self.size)

    def get_cell(self, row: int, col: int):
        """
        Retrieves the cell with given position
        :param row:
        :param col:
        :return: Cell
        """
        return self.grid[col, row]

    def get_board_grid(self):
        return self.grid

    def get_cells_as_list(self):
        cells = []
        for row in self.get_board_grid():
            for col in row:
                if type(col) == Cell:
                    cells.append(col)
        return cells

    def get_open_cells(self):
        cells = []
        for row in self.get_board_grid():
            for col in row:
                if type(col) == Cell:
                    if col.is_empty:
                        cells.append(col)
        return cells
