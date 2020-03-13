import numpy as np
from Project1.node import Node


class Board:
    """
    Creates a board for the game given parameters as size and shape.
    """
    def __init__(self, size, shape):
        """
        Set parameters for shape and size
        :param size: Size of the board (i.e., including all numbers in [3, 10])
        :param shape: Shape of the board (diamond or triangle)
        """
        self.size = size
        self.type = shape.lower()
        # grid is represented with (row, col)
        self.grid = np.ndarray(shape=(size, size), dtype=Node)
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
                self.grid[r, c] = Node(c, r, s)
                s += 1

        # List of potential neighbours for each node, board shape restrictions are handled in the Node class.
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
                self.grid[r, c] = Node(c, r, s)
                s += 1
        # List of potential neighbours for each node, board shape restrictions are handled in the Node class.
        # After a transformation from equilateral triangle to array the neighbours are represented as
        # [NW, W, N, S, E, SE].
        neighbours = [[-1, -1], [-1, 0], [0, -1], [0, 1], [1, 0], [1, 1]]
        for r in range(self.size):
            for c in range(r + 1):
                self.grid[r, c].add_neighbours(neighbours, self.grid, self.size)

    def get_node(self, x, y):
        """
        Retrieves desired node depending on position
        :param x: row of node
        :param y: column of node
        :return:
        """
        return self.grid[x, y]

    def move(self, from_node, to_node):
        """
        Performs the move of a peg on the board. Contains validity check of the entered move.
        :param from_node: node to be jumped from
        :param to_node: node being jumped over
        :return: True or False depending on whether the move was successful or not
        """
        row = from_node.x + 2 * (to_node.x - from_node.x)
        col = from_node.y + 2 * (to_node.y - from_node.y)
        # Restrictions for legal move, i.e., no moves outside the board and to_node (the one to be jumped over) must not
        # be empty, while the node jumped to must be.
        if self.size > row >= 0 and self.size > col >= 0 and not to_node.is_empty and self.get_node(row, col).is_empty:
            return self.get_node(row, col)
        else:
            return False
