import networkx as nx
import matplotlib.pyplot as plt
from board import Board


class Visualize:
    def __init__(self, board: Board, delay):
        self.delay = delay
        self.board = board
        self.graph = nx.Graph()
        self.pos = dict()
        self.init_graph()

    def init_graph(self):
        for row in range(self.board.size):
            for col in range(self.board.size):
                cell = self.board.get_cell(col, row)
                if cell is None:
                    continue
                if self.board.type == 't':
                    self.pos[cell] = (-10 * row + 20 * col, -10 * row)
                else:
                    self.pos[cell] = (-10 * row + 10 * col, -20 * row - 20 * col)
                self.graph.add_node(cell)
                for neighbour in cell.neighbours:
                    self.graph.add_edge(cell, neighbour)
        self.update_graph(None, (None, None))

    def update_graph(self, to_cell, action):
        fig = plt.figure(figsize=(9, 7))
        plt.axes()
        if to_cell is not None:
            self.graph.add_node(to_cell)
            row = to_cell.y
            col = to_cell.x
            if self.board.type == 't':
                self.pos[to_cell] = (-10 * row + 20 * col, -10 * row)
            else:
                self.pos[to_cell] = (-10 * row + 10 * col, -20 * row - 20 * col)
        for cell in self.graph:
            if cell.is_empty:
                if cell == action[0]:
                    nx.draw(self.graph,
                            pos=self.pos,
                            nodelist=[cell],
                            node_color='black',
                            node_size=200,
                            ax=fig.axes[0])
                else:
                    nx.draw(self.graph,
                            pos=self.pos,
                            nodelist=[cell],
                            node_color='black',
                            node_size=800,
                            ax=fig.axes[0])
            else:
                if cell is to_cell:
                    nx.draw(self.graph,
                            pos=self.pos,
                            nodelist=[cell],
                            node_color='blue',
                            node_size=2400,
                            ax=fig.axes[0])
                else:
                    nx.draw(self.graph,
                            pos=self.pos,
                            nodelist=[cell],
                            node_color='blue',
                            node_size=800,
                            ax=fig.axes[0])
        if self.delay:
            plt.show(block=False)
            plt.pause(self.delay)
            plt.close()
        else:
            plt.show(block=True)


# TODO: implement method for updating data instead of re-plotting for each new board (state)


if __name__ == '__main__':
    brd = Board(5, 't', [[2, 1]])
    vis = Visualize(brd, 0)
