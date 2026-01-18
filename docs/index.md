Deep Q-Learning improves upon Q-learning, in the way that we can now store a continuous state-space, which is crucial for finance since most often we keep track of the returns/volatility of the past X-amount of days/weeks/months.

Read eg.: https://www.geeksforgeeks.org/deep-learning/deep-q-learning/

In my implementation, the state will be a tensor of shape (batch_size, n_features), and the action space will be the set of possible weight combinations. n_features will be the amount of assets we are considering and batch_size an integer we choose freely.

The neural network will approximate the Q-function. It will take the state tensor as input and return an array of Q-values as its output, one for each action.

In a better version of Deep Q-Learning, we keep track of two separate neural networks: a target- and a train-neural network.

The target network is used for evaluation, that is, to sample the actual Q-values, while the main network is used for picking the next best action. Gradient descent is performed on the main network to constantly updating its parameters, while the weights of the target network remain mostly stationary and are periodically updated to keep it in-sync with the parameters of the main network.

Keeping track of two NN's like this prevents using the same function to both evaluate and choose the next action, which stimulates overfitting because it is directly dependent on the current set of parameters.

By keeping the main network's parameters mostly stationary we are solving a concrete gradient-descent problem with a given (stationary) set of parameters.