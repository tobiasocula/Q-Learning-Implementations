import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf

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
        


def construct_dqn_1(
        n_actions, input_shape
):
    model = tf.keras.models.Sequential()
    # Conv tf.keras.layers: for time series/financial data, use 1D conv (across time)
    model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))  # 1st layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))  # 2nd layer
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))  # 3rd layer
    model.add(tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'))  # 4th layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))  # Fully connected for Q-value learning
    model.add(tf.keras.layers.Dense(n_actions, activation=None))  # Output Q-values for all actions
    return model

def construct_dqn_2(
        n_actions, input_shape
):
    model = tf.keras.models.Sequential()
    # Add LSTM recurrent layer
    model.add(tf.keras.layers.LSTM(64, input_shape=input_shape))
    # Add dense layer(s) for Q-value mapping
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation=None)) # Q-values for all actions
    return model

def construct_dqn_3(
        n_actions, input_shape
):
    model = tf.keras.models.Sequential()
    # Add GRU recurrent layer
    model.add(tf.keras.layers.GRU(64, input_shape=input_shape))
    # Add dense layer(s) for Q-value mapping
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(n_actions, activation=None)) # Q-values for all actions
    return model