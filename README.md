A repository where I learn and implement various reinforcement learning algorithms.

### Q-Learning

In this algorithm, we have the sets $S$ and $A$, being the state and action space respectively. The goal is to keep track of a so-called Q-table, which will store the Q-values for all possible actions and states. The algorithm will find the best possible action to take for each state, meaning the action that maximizes the Q-value. One may see the Q-value as representing a kind of mix between the immediate reward and the future reward.

Read more: https://en.wikipedia.org/wiki/Q-learning

In my implementation (in the folder QLearning), I implement this with the state space being the set of possible financial regimes, determined by the volatility of returns. The action space will be the different asset weigts one can have in the portfolio.

### Deep Q-Learning

In Deep Q-Learning, we use a neural network instead of a Q-table to keep track of the Q-values for each action and state.

This allows us to have a continuous state space instead of the previously limited discrete state space.

In my implementation, the state will be a tensor of shape (batch_size, n_features), and the action space will again be the set of possible weight combinations.

The neural network will approximate the Q-function. It will take the state tensor as input and return an array of Q-values as its output, one for each action.