import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
import pandas as pd

"""

contains code for all applications made within this folder, being classes and functions.

"""

class Tester:
    def __init__(self, *model_args):
        self.models = []
        for ma in model_args:
            model = Model2(*ma)
            self.models.append(model)
    def train_all(self):
        total_losses = []
        total_avg_q_values = []
        total_reward_list = []
        for ma in self.models:
            ma.train()
            total_losses.append(ma.losses)
            total_avg_q_values.append(ma.avg_q_values)
            total_reward_list.append(ma.reward_list)
        self.total_losses = total_losses
        self.total_avg_q_values = total_avg_q_values
        self.total_reward_list = total_reward_list

    def show_stats(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3)
        for i, (l, a, r) in enumerate(zip(self.total_losses, self.total_avg_q_values, self.total_reward_list)):
            ax1.plot(l, label=f"model {i}")
            ax2.plot(a, label=f"model {i}")
            ax3.plot(r, label=f"model {i}")
        ax1.set_ylabel("losses")
        ax2.set_ylabel("average Q values")
        ax3.set_ylabel("rewards")
        plt.show()

    def save_all(self, dirname):
        if not all([m.trained for m in self.models]):
            raise AssertionError()
        
        for i, m in enumerate(self.models):
            m.save(dirname=dirname, modelname=f"model_{i}")

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.len = 0
        self.pos = 0
    def sample(self, size):
        # returns random sequence of lists (state, reward, action, next_state)
        if self.len < size:
            return self.buffer
        indices = np.random.choice(self.len, size=size, replace=False)
        return [self.buffer[i] for i in indices]

    def update(self, state, reward, action, next_state, done):
        if self.len < self.capacity:
            self.buffer.append([state, reward, action, next_state, done])
            self.len += 1
        else:
            self.buffer[self.pos] = [state, reward, action, next_state, done]
        self.pos = (self.pos + 1) % self.capacity
        
    def categorize_samples(self, samples):
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for s in samples:
            states.append(s[0])
            rewards.append(s[1])
            actions.append(s[2])
            next_states.append(s[3])
            dones.append(s[4])
        return np.array(states), np.array(rewards), np.array(actions), np.array(next_states), np.array(dones)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.len = 0
        self.pos = 0
    def sample(self, size):
        # returns random sequence of lists (state, reward, action, next_state)
        if self.len < size:
            return self.buffer
        indices = np.random.choice(self.len, size=size, replace=False)
        return [self.buffer[i] for i in indices]

    def update(self, state, reward, action, next_state, done):
        if self.len < self.capacity:
            self.buffer.append([state, reward, action, next_state, done])
            self.len += 1
        else:
            self.buffer[self.pos] = [state, reward, action, next_state, done]
        self.pos = (self.pos + 1) % self.capacity
        
    def categorize_samples(self, samples):
        states = []
        rewards = []
        actions = []
        next_states = []
        dones = []
        for s in samples:
            states.append(s[0])
            rewards.append(s[1])
            actions.append(s[2])
            next_states.append(s[3])
            dones.append(s[4])
        return np.array(states), np.array(rewards), np.array(actions), np.array(next_states), np.array(dones)

class Model1:
    def __init__(self,
        state_space,
        rep_buffer_size,
        rb_sample_size,
        action_space,
        gamma,
        training_lookback_period = 5,
        t_replay_buffer = 30
        ):
        self.gamma = gamma
        self.action_space = action_space
        self.n_assets, self.n_actions = action_space.shape
        self.training_lookback_period = training_lookback_period
        self.state_space = state_space
        self.cov = np.cov(self.state_space, rowvar=False)
        self.T, self.features_dim = self.state_space.shape
        self.t_replay_buffer = t_replay_buffer
        self.replay_buffer = ReplayBuffer(rep_buffer_size)
        self.model = construct_dqn_1(self.n_actions, (self.training_lookback_period, self.features_dim))
        self.model.compile(optimizer='adam', loss='mse')
        self.state = None
        self.rb_sample_size = rb_sample_size

    def step(self, action, t):
        weights = self.action_space[:, action]
        ret = np.dot(weights, self.state_space[t, :])
        portfolio_variance = weights.T @ self.cov @ weights
        reward = ret / np.sqrt(portfolio_variance)
        next_state = self.state_space[t - self.training_lookback_period:t, :]
        return next_state, reward, t == self.T

    def train(self):
        losses = []
        avg_q_values = []
        reward_list = []
        for t in range(self.training_lookback_period, self.T):
            subset = self.state_space[t - self.training_lookback_period:t, :]
            subset = np.expand_dims(subset, axis=0)
            Q_values = self.model.predict(subset) # size (n_actions)
            action = np.argmax(Q_values)
            
            # determine next action, reward, done and state
            new_state, reward, done = self.step(action, t)
            reward_list.append(reward)

            self.replay_buffer.update(self.state, reward, action, new_state, done)
            self.state = new_state
            if t >= self.t_replay_buffer:
                # sample from replay buffer
                samples = self.replay_buffer.sample(size=self.rb_sample_size) # array of length rb_sample_size
                # one (state, reward, action, next_state, done) per action
                states, rewards, actions, next_states, dones = self.replay_buffer.categorize_samples(samples)
                current_Q = self.model.predict_on_batch(states) # shape: (batch, n_actions)
                avg_q_values.append(np.mean(current_Q, axis=1))
                next_Q = self.model.predict_on_batch(next_states) # shape: (batch, n_actions)
                max_next_Q = np.max(next_Q, axis=1) # shape: (batch)
                target_Q = current_Q.copy() # shape (rb_sample_size, n_actions)
                for i in range(self.rb_sample_size):
                    target_Q[i, actions[i]] = rewards[i] + self.gamma * (0 if dones[i] else 1) * max_next_Q[i]
                # train model on one batch
                loss = self.model.train_on_batch(states, target_Q)
                losses.append(loss)
        return losses, avg_q_values, reward_list

class Model2:
    def __init__(self,
        state_space,
        rb_capacity,
        rb_sample_size,
        action_space,
        gamma,
        target_update_freq,
        training_lookback_period,
        t_replay_buffer,
        model_constructor: callable
        ):
        self.gamma = gamma
        self.action_space = action_space
        self.n_assets, self.n_actions = action_space.shape
        self.training_lookback_period = training_lookback_period
        self.state_space = state_space
        self.state = self.state_space[:training_lookback_period, :]
        self.cov = np.cov(self.state_space, rowvar=False)
        self.T, self.features_dim = self.state_space.shape
        self.t_replay_buffer = t_replay_buffer
        self.replay_buffer = ReplayBuffer(rb_capacity)
        self.main_model = model_constructor(self.n_actions, (self.training_lookback_period, self.features_dim))
        self.main_model.compile(optimizer='adam', loss='mse')
        self.target_model = model_constructor(self.n_actions, (self.training_lookback_period, self.features_dim))
        self.target_model.set_weights(self.main_model.get_weights())
        self.batch_size = rb_sample_size
        self.target_update_freq = target_update_freq

        self.trained = False

    def step(self, action, t):
        weights = self.action_space[:, action]
        ret = np.dot(weights, self.state_space[t, :])
        portfolio_variance = weights.T @ self.cov @ weights
        reward = ret / np.sqrt(portfolio_variance)
        next_state = self.state_space[t - self.training_lookback_period:t, :]
        return next_state, reward, t == self.T

    def train(self):
        step = 0
        losses = []
        avg_q_values = []
        reward_list = []
        for t in range(self.training_lookback_period, self.T):
            subset = self.state_space[t - self.training_lookback_period:t, :]
            subset = np.expand_dims(subset, axis=0)
            Q_values = self.main_model.predict(subset) # size (n_actions)
            action = np.argmax(Q_values)
            
            # determine next action, reward, done and state
            new_state, reward, done = self.step(action, t)
            reward_list.append(reward)
            
            self.replay_buffer.update(self.state, reward, action, new_state, done)
            self.state = new_state
            if t >= self.t_replay_buffer:
                # sample from replay buffer
                samples = self.replay_buffer.sample(size=self.batch_size) # array of length rb_sample_size
                states, rewards, actions, next_states, dones = self.replay_buffer.categorize_samples(samples)

                current_Q = self.main_model.predict_on_batch(states) # (batch_size, n_actions)
                avg_q_values.append(np.mean(current_Q, axis=1))

                next_Q_main = self.main_model.predict_on_batch(next_states) # (batch_size, n_actions)
                next_Q_target = self.target_model.predict_on_batch(next_states) # (batch_size, n_actions)

                actions = np.argmax(next_Q_main, axis=1) # (batch_size)
                actual_Q = next_Q_target[np.arange(self.batch_size), actions] # (batch_size)

                target_Q = next_Q_main.copy()
                
                for i in range(self.batch_size):
                    target_Q[i, actions[i]] = rewards[i] + self.gamma * (0 if dones[i] else 1) * actual_Q[i]
                # train model on one batch
                loss = self.main_model.train_on_batch(states, target_Q)
                losses.append(loss)
                if step % self.target_update_freq == 0:
                    self.target_model.set_weights(self.main_model.get_weights())

            step += 1

        self.losses = losses
        self.avg_q_values = avg_q_values
        self.reward_list = reward_list

        self.trained = True

    def save(self, dirname: Path, modelname: str):

        if not self.trained:
            raise AssertionError()

        Path(dirname).mkdir(parents=True, exist_ok=True)
        self.main_model.save(Path(dirname) / f"{modelname}.keras")

class ModelDeployer:
    def __init__(self, modelpath, data, action_space):
        """data is a T x (n_features) matrix (return matrix)
        action_space is a (n_features) x (n_actions) matrix"""
        self.model = tf.keras.models.load_model(modelpath)
        self.data = data
        self.action_space = action_space

    def run(self, period):
        if period > self.data.shape[0]:
            raise AssertionError()
        
        return_sequence = np.empty(self.data.shape[0] - period)
        for t in range(period, self.data.shape[0]):
            subset = self.data[t - period:t, :]
            Q_values = self.model.predict(subset)
            action_idx = np.argmax(Q_values)
            weights = self.action_space[:, action_idx]
            result_return = np.dot(weights, self.data[t, :])
            return_sequence[t] = result_return

        self.returns = return_sequence
        self.cum_return = np.cumprod(self.returns + 1) - 1
        self.returns_vola_ann = pd.Series(self.returns).rolling(252, min_periods=1).std().to_numpy()
        self.sharpe = 255 * np.mean(self.cum_return) / (np.std(self.returns) * np.sqrt(255))
        self.sortino = 255 * np.mean(self.cum_return) / (np.std(self.returns[self.returns < 0]) * np.sqrt(255))

        r = self.data @ np.full(self.data.shape[1], 1 / self.data.shape[1]).T
        self.returns_equal_weights = np.cumprod(r + 1) - 1
    
    def plot(self):
        plt.plot(self.cum_return, label="model performance")
        plt.plot(self.returns_equal_weights, label="equal weights performance")
        plt.show()

def construct_dqn_1(n_actions, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 3, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_actions)(x)  # No activation for raw Q-values
    model = tf.keras.models.Model(inputs, outputs)
    return model

def construct_dqn_2(n_actions, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.LSTM(64)(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_actions)(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

def construct_dqn_3(n_actions, input_shape):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.GRU(64)(inputs)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(n_actions)(x)
    model = tf.keras.models.Model(inputs, outputs)
    return model

def construct_pacman_model():
    # Suppose input channels encode: {wall, pellet, power-pellet, Pac-Man, ghosts}
    # Update shape as needed: (height, width, channels)
    inputs = tf.keras.layers.Input(shape=(19, 19))

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(4)(x)  # 4 actions: up, down, left, right

    model = tf.keras.models.Model(inputs, outputs)
    return model


class PacmanModel:
    def __init__(self):
        pass