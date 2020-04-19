import torch
import torch.nn as nn
import numpy as np


class ANET(nn.Module):
    def __init__(self, io_dim, H_dims, learning_rate, optimizer, activation_fn, epochs):
        super(ANET, self).__init__()
        self.alpha = learning_rate
        self.epochs = epochs
        activation_fn = self.__choose_activation_fn(activation_fn)
        layers = self.gen_layers(io_dim, H_dims, activation_fn)
        self.model = torch.nn.Sequential(*layers)
        self.loss_fn = torch.nn.BCELoss(reduction="mean")
        self.optimizer = self.__choose_optimizer(list(self.model.parameters()), optimizer)

    def gen_layers(self, io_dim, H_dims, activation_fn):
        layers = [torch.nn.Linear(io_dim + 1, H_dims[0]), torch.nn.Dropout(0.5)]
        layers.append(activation_fn) if activation_fn is not None else None
        for i in range(len(H_dims)-1):
            layers.append(torch.nn.Linear(H_dims[i], H_dims[i+1]))
            layers.append(torch.nn.Dropout(0.5))
            layers.append(activation_fn) if activation_fn is not None else None
        layers.append(torch.nn.Linear(H_dims[-1], io_dim))
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

    def get_distribution(self, state, legal_moves):
        factor = [1 if move in legal_moves else 0 for move in state.initial_moves]
        input = self.transform(state.repr_state)
        probs = self.forward(input).data.numpy()
        sum = 0
        new_probs = np.zeros(state.size ** 2)
        for i in range(state.size ** 2):
            if factor[i]:
                sum += probs[i]
                new_probs[i] = probs[i]
            else:
                new_probs[i] = 0
        new_probs /= sum
        indices = np.arange(state.size ** 2)
        stochastic_index = np.random.choice(indices, p=new_probs)
        greedy_index = np.argmax(new_probs)
        return new_probs, stochastic_index, greedy_index

    def get_status(self, input, target):
        x = self.transform(input)
        y = self.transform(target)
        pred_y = self.forward(x)
        loss = self.loss_fn(pred_y, y)
        acc = pred_y.argmax(dim=1).eq(y.argmax(dim=1)).sum().numpy() / len(y)
        return loss.item(), acc

    def save(self, size, level):
        torch.save(self.state_dict(), f"models/{size}_ANET_level_{level}")
        print(f"\nModel has been saved to models/{size}_ANET_level_{level}")

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
    def __choose_activation_fn(activation_fn):
        return {
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "sigmoid": torch.nn.Sigmoid(),
            "linear": None
        }[activation_fn]
