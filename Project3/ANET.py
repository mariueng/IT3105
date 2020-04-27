import torch
import torch.nn as nn
import numpy as np


class ANET(nn.Module):
    def __init__(self, io_dim, hidden_layers, learning_rate, optimizer, activation, epochs):
        super(ANET, self).__init__()
        self.alpha = learning_rate
        self.epochs = epochs
        activation = self.__choose_activation_function(activation)
        layers = self.gen_layers(io_dim, hidden_layers, activation)
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

    def gen_layers(self, io_dim, hidden_layers, activation_fn):
        layers = [torch.nn.Linear(io_dim + 1, hidden_layers[0])]
        layers.append(torch.nn.Dropout(0.5))
        layers.append(activation_fn) if activation_fn is not None else None
        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(torch.nn.Dropout(0.5))
            layers.append(activation_fn) if activation_fn is not None else None
        layers.append(torch.nn.Linear(hidden_layers[-1], io_dim))
        layers.append(torch.nn.Softmax(dim=-1))
        return layers

    @staticmethod
    def transform(data):
        return torch.FloatTensor(data)

    def fit(self, input_, target):
        x = self.transform(input_)
        y = self.transform(target)
        for i in range(self.epochs):
            pred_y = self.model(x)
            loss = self.loss_fn(pred_y, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy()/len(y)
        return loss.item(), acc

    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    def get_distribution(self, state):
        legal_moves = state.get_legal_actions()
        factor = [1 if move in legal_moves else 0 for move in state.initial_moves]
        input_ = self.transform(state.repr_state)
        probabilities = self.forward(input_).data.numpy()
        sum_ = 0
        new_probabilities = np.zeros(state.size ** 2)
        for i in range(state.size ** 2):
            if factor[i]:
                sum_ += probabilities[i]
                new_probabilities[i] = probabilities[i]
            else:
                new_probabilities[i] = 0
        new_probabilities /= sum_
        indices = np.arange(state.size ** 2)
        stochastic_index = np.random.choice(indices, p=new_probabilities)
        greedy_index = np.argmax(new_probabilities)
        return new_probabilities, stochastic_index, greedy_index

    def get_status(self, input_, target):
        x = self.transform(input_)
        y = self.transform(target)
        pred_y = self.forward(x)
        loss = self.loss_fn(pred_y, y)
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy() / len(y)
        return loss.item(), acc

    def save(self, size, level):
        torch.save(self.state_dict(), f"models/{size}_ANET_level_{level}")
        print(f"Model has been saved to models/{size}_ANET_level_{level}")

    def load(self, size, level):
        self.load_state_dict(torch.load(f"models/{size}_ANET_level_{level}"))
        print(f"Loaded model from models/{size}_ANET_level_{level}")

    def __choose_optimizer(self, params, optimizer):
        return {
            "Adagrad": torch.optim.Adagrad(params, lr=self.alpha),
            "SGD": torch.optim.SGD(params, lr=self.alpha),
            "RMSprop": torch.optim.RMSprop(params, lr=self.alpha),
            "Adam": torch.optim.Adam(params, lr=self.alpha)
        }[optimizer]

    @staticmethod
    def __choose_activation_function(activation_fn):
        return {
            "ReLU": nn.ReLU(),
            "Tanh": nn.Tanh(),
            "Sigmoid": nn.Sigmoid(),
            "Linear": nn.Identity(),
            "LeakyReLU": nn.LeakyReLU()
        }[activation_fn]
