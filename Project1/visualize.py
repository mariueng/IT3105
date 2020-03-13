import networkx as nx
import matplotlib.pyplot as plt
from Project1.board import Board


def init_graph(board):
    graph = nx.Graph()
    pos = dict()
    for row in range(board.size):
        for col in range(board.size):
            node = board.get_node(row, col)
            if node is None:
                continue
            if board.type == 't':
                pos[node] = (-10 * row + 20 * col, -10 * row)
            else:
                pos[node] = (-10*row + 10 * col, -20 * row - 20 * col)
            graph.add_node(node)
            print("Node: " + str(node))
            print("Neighbours: " + str(node.neighbours))
            for neighbour in node.neighbours:
                graph.add_edge(node, neighbour)
    ax = plt.gca()
    options = dict(node_color='black',
                   font_color='white',
                   node_size=1000,
                   ax=ax)
    nx.draw(graph, pos, with_labels=True, **options)
    print(graph.nodes)
    plt.show()


if __name__ == '__main__':
    brd = Board(3, 'd')
    print(brd.grid)
    init_graph(brd)
