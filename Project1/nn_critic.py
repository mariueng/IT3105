import math
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class NNCritic:
    def __init__(self, learning_rate, decay_rate_critic, discount_factor, input_dim, hidden_layers):
        self.learning_rate = learning_rate
        self.decay_rate_critic = decay_rate_critic
        self.discount_factor = discount_factor
        self.eligibilities = []
        # Build model
        self.model = Sequential()
        self.model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
        for hidden_layer in hidden_layers:
            self.model.add(Dense(hidden_layer, activation='relu', kernel_initializer="uniform"))
        self.model.add(Dense(1, activation='linear'))
        self.reset_eligibilities()
        sgd = tf.keras.optimizers.Adagrad(learning_rate=self.learning_rate)
        self.model.compile(optimizer=sgd, loss=tf.keras.losses.MeanSquaredError(), run_eagerly=True)

    def reset_eligibilities(self):
        self.eligibilities.clear()
        for params in self.model.trainable_weights:
            self.eligibilities.append(tf.zeros_like(params))

    def update_all_eligibilities(self):
        tensor = tf.convert_to_tensor(self.decay_rate_critic * self.discount_factor, dtype=tf.dtypes.float32)
        for i in range(len(self.eligibilities)):
            self.eligibilities[i] = tf.multiply(tensor, self.eligibilities[i])

    def get_value_state(self, state):
        state = [tf.strings.to_number(b, out_type=tf.dtypes.int32) for b in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        return self.model(state).numpy()[0][0]

    def compute_td_error(self, reinforcement, old_state, state):
        target = reinforcement + self.discount_factor * self.get_value_state(state)
        td_error = target - self.get_value_state(old_state)
        return td_error

    def modify_gradients(self, gradients, td_error):
        for j in range(len(gradients)):
            self.eligibilities[j] = tf.add(self.eligibilities[j], gradients[j])
            gradients[j] = self.eligibilities[j] * td_error
        return gradients

    def fit(self, reinforcement, old_state, state, td_error):
        with tf.GradientTape() as tape:
            old_state, state, gamma, reinforcement = self.convert_data(old_state, state, self.discount_factor,
                                                                       reinforcement)
            target = tf.add(reinforcement, tf.multiply(gamma, self.model(state, training=True)))
            prediction = self.model(old_state, training=True)
            loss = self.model.loss(target, prediction)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        modified_gradients = self.modify_gradients(gradients, td_error)
        self.model.optimizer.apply_gradients(zip(modified_gradients, self.model.trainable_variables))

    # TODO: Remove discount_factor as parameter???
    def convert_data(self, old_state, state, discount_factor, reinforcement):
        old_state = [tf.strings.to_number(b, out_type=tf.dtypes.float32) for b in old_state]  # convert to array
        old_state = tf.convert_to_tensor(np.expand_dims(old_state, axis=0))
        state = [tf.strings.to_number(b, out_type=tf.dtypes.float32) for b in state]
        state = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        discount_factor = tf.convert_to_tensor(self.discount_factor, dtype=tf.dtypes.float32)
        reinforcement = tf.convert_to_tensor(reinforcement, dtype=tf.dtypes.float32)
        return old_state, state, discount_factor, reinforcement
