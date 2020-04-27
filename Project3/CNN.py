import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from state_manager import Hex


class CNN(nn.Module):

    def __init__(self, io_dim, hidden_layers=(32, 32), learning_rate=0.01, optimizer='Adam', activation='ReLU', epochs=10):
        """
        Initializes Convolutional Neural Network (CNN)
        :param io_dim: Dimension of input and out layers
        :param hidden_layers: List of hidden layers: e.g. [32, 32, ...]
        :param learning_rate: Learning rate for CNN
        :param epochs: Epochs to train CNN on
        :param activation: Activation function
        :param optimizer: Optimizer for CNN
        """
        super(CNN, self).__init__()
        # Helper state to create input dimensions for passes
        # TODO: See if this can be removed?
        self.game = Hex(int(np.sqrt(io_dim)))

        self.io_dim = io_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        h_layers = self.build_model(hidden_layers, activation)
        self.model = nn.Sequential(h_layers)
        params = list(self.parameters())
        self.optimizer = self.__choose_optimizer(params, optimizer)
        self.loss_fn = nn.CrossEntropyLoss()

    def build_model(self, hidden_layers, activation):
        """
        Builds model with specified structure
        :param hidden_layers: List of hidden layers
        :param activation: Activation function to be used in all layers
        :return: Ordered Dictionary of layers
        """
        layers = OrderedDict([
            ('0', nn.ZeroPad2d(2)),
            ('1', nn.Conv2d(9, hidden_layers[0], 3)),
            ('2', self.__choose_activation_function(activation))])
        for i in range(len(hidden_layers) - 1):
            layers[str(len(layers))] = nn.ZeroPad2d(1)
            layers[str(len(layers))] = nn.Conv2d(hidden_layers[i], hidden_layers[i + 1], 3)
            layers[str(len(layers))] = self.__choose_activation_function(activation)
        layers[str(len(layers))] = nn.Conv2d(hidden_layers[-1], 1, 3)
        layers[str(len(layers))] = self.__choose_activation_function(activation)
        layers[str(len(layers))] = nn.Conv2d(1, 1, 1)
        return layers

    def forward(self, x, training=False):
        """
        Computes output tensors from input (which is transformed to tensors)
        :param x: input (not tensor)
        :param training: Boolean, True if the model should be trained
        :return: Output tensor with softmax as activation function
        """
        self.train(training)
        x = self.transform_input(x)
        x = self.model(x)
        x = x.reshape(-1, self.io_dim)
        return F.softmax(x, dim=1)

    # TODO: Check log-probabilities of 180-degree rotation
    '''
    def log_prob(self, x, training=False):
        
    '''

    def raw_forward(self, x, training=False):
        """
        Computes output tensors from input (which is transformed to tensors)
        :param x: input (not tensor)
        :param training: Boolean, True if the model should be trained
        :return: (Raw) Output tensor
        """
        self.train(training)
        x = self.transform_input(x)
        x = self.model(x)
        return x.reshape(-1, self.io_dim)

    def fit(self, x, y):
        """
        Fits the data with the existing CNN
        :param x: Input
        :param y: Target
        :return: Loss, accuracy
        """
        y = torch.FloatTensor(y)
        for i in range(self.epochs):
            pred_y = self.raw_forward(x, training=True)
            loss = self.loss_fn(pred_y, y.argmax(dim=1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy() / len(y)
        return loss.item(), acc

    def evaluate(self, x, y):
        """
        Evaluates the loss and accuracy of the CNN with provided input and targets
        :param x: Input values
        :param y: Target values
        :return: loss, accuracy
        """
        y = torch.FloatTensor(y)
        pred_y = self.raw_forward(x)
        loss = self.loss_fn(pred_y, y.argmax(dim=1))
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy() / len(y)
        return loss.item(), acc

    def transform_input(self, input_):
        """
        Transforms flat game state into 9 input planes (size, size):
        - Empty/p1/p2       0/1/2   (empty/red/black)
        - To play           3/4     (p1/p2 to play)
        - P1 bridge         5       (red bridge endpoints)
        - P2 bridge         6       (black bridge endpoints)
        - To play bridge    7       (active if cell is a form bridge)
        - To play bridge    8       (active if cell is a save bridge)
        :param input_: input data
        :return: transformed vector that fits neural network
        """
        size = int(np.sqrt(self.io_dim))
        out = []
        for x in input_:
            player = int(x[0])
            x = x[1:].reshape(size, size)
            planes = np.zeros(9 * size ** 2).reshape(9, size, size)
            planes[player + 2] += 1  # plane 3/4
            for row in range(size):
                for col in range(size):
                    piece = int(x[row][col])
                    planes[piece][row][col] = 1  # plane 0-2
                    if (row, col) in self.game.bridge_neighbours:
                        for (row_bridge, col_bridge) in self.game.bridge_neighbours[(row, col)]:
                            if piece == 0:
                                if x[row_bridge][col_bridge] == player:
                                    planes[7][row][col] = 1  # 7: form bridge
                            else:
                                if x[row_bridge][col_bridge] == piece:
                                    planes[piece + 4][row][col] = 1  # 5/6: bridge endpoints
                                    cn = list(set(self.game.neighbours[(row, col)]).intersection(
                                        set(self.game.neighbours[(row_bridge, col_bridge)])))  # common neighbors
                                    r1, c1 = cn[0]
                                    r2, c2 = cn[1]
                                    if x[r1][c1] == 0 and x[r2][c2] == 3 - player:
                                        planes[8][r1][c1] = 1
                                    elif x[r2][c2] == 0 and x[r1][c1] == 3 - player:
                                        planes[8][r2][c2] = 1
            out.append(planes)
        return torch.FloatTensor(out)

    def get_distribution(self, state):
        """
        Returns the distribution of the current CNN
        :param state: Current game state
        :return: probability distribution, stochastic index, greedy index
        """
        legal_moves = state.get_actions()
        factor = [1 if move in legal_moves else 0 for move in state.initial_moves]
        probabilities = self.forward([state.repr_state]).data.numpy()[0]
        sum = 0
        new_probs = np.zeros(state.size ** 2)
        for i in range(state.size ** 2):
            if factor[i]:
                sum += probabilities[i]
                new_probs[i] = probabilities[i]
            else:
                new_probs[i] = 0
        new_probs /= sum
        indices = np.arange(state.size ** 2)
        stochastic_index = np.random.choice(indices, p=new_probs)
        greedy_index = np.argmax(new_probs)
        return new_probs, stochastic_index, greedy_index

    def save(self, size, level):
        """
        Saves the current CNN
        :param size: size of board the CNN is trained on
        :param level: Episode when CNN is saved
        :return:
        """
        torch.save(self.state_dict(), f"models/{size}_ANET_level_{level}")
        print(f"Model has been saved to models/{size}_ANET_level_{level}")

    def load(self, size, level):
        """
        Loads CNN with size and level
        :param size: size of the board the CNN is trained on
        :param level: Episode when CNN was saved
        :return:
        """
        self.load_state_dict(torch.load(f"models/{size}_ANET_level_{level}"))
        print(f"Loaded model from models/{size}_ANET_level_{level}")

    def __choose_optimizer(self, params, optimizer):
        """
        Chooses optimizer
        :param params: Parameters of CNN
        :param optimizer: Chosen optimizer
        :return: Optimizer
        """
        return {
            "Adagrad": torch.optim.Adagrad(params, lr=self.learning_rate),
            "SGD": torch.optim.SGD(params, lr=self.learning_rate),
            "RMSprop": torch.optim.RMSprop(params, lr=self.learning_rate),
            "Adam": torch.optim.Adam(params, lr=self.learning_rate)
        }[optimizer]

    @staticmethod
    def __choose_activation_function(activation_fn):
        """
        Helper method for choosing activation function
        :param activation_fn: Chosen activation function
        :return: activation function
        """
        return {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Linear": nn.Identity(),
            "LeakyReLU": nn.LeakyReLU()
        }[activation_fn]
