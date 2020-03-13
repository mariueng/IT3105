class Node:
    """
    Creates a node representing a single cell on the board
    """
    def __init__(self, x, y, s):
        """
        Initializes node with position
        :param x: row position
        :param y: column position
        """
        self.x = x
        self.y = y
        self.is_empty = False
        self.neighbours = []
        # Only for testing purposes
        self.name = s

    # Method for obtaining name during testing
    def get_name(self):
        return self.name

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_pos(self):
        return [self.x, self.y]

    def get_neighbours(self):
        return self.neighbours

    # Descr.
    def add_neighbours(self, directions, grid, size):
        for dir in directions:
            if 0 <= self.x + dir[0] < size and 0 <= self.y + dir[1] < size:
                neighbour = grid[self.y + dir[1], self.x + dir[0]]
                if not self.is_neighbour(neighbour):
                    if neighbour is not None:
                        self.neighbours.append(neighbour)

    def is_neighbour(self, node):
        # NB check if result is a truth array instead of true or false
        return self.neighbours.__contains__(node)

    def remove_peg(self):
        self.is_empty = True

    def add_peg(self):
        self.is_empty = False

    def __repr__(self):
        return str(self.name)


if __name__ == '__main__':
    q = 3
