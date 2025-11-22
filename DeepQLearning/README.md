### Deep Q-Learning

From the Q-Learning algorithm, we now go over the deep Q learning algorithm, where instead of using a Q-table to store Q-values directly, we store and use a neural network to approximate the Q-function.

Instead of the state space being a discrete array of possible values, now the state space can be continuous and more dynamic. The state space here will represent the returns of the past few days (determined by a hyperparameter). The action space will still be discrete, being possible values for the asset weights.

#### Files

##### simple_dqn.ipynb

The first implementation of a DQN model. We use a single DQN instance, to both generate and predict Q-values and update the network accordingly.

Here we use a singular DQN instance and a replay buffer.

##### using_target_network.ipynb

A second implementation, where we use two separate DQN instances and a so-called Dual DQN strategy.

##### constructing_models.py

This file contains classes and functions for constructing and training DQN models.

The Model1 and Model2 classes implement the actual DQN algorithm, using a keras neural network under the hood. Model2 is more extensive, with the ability to pass a custom construction function for the model, and is used in the target network implementation.

The Tester class is meant to store multiple models (Model2 instances in this case), and test and compare all passed models.

The ModelDeployer class loads a model class and deploys it on evaluation data. It can also plot statistics of the run.

The ReplayBuffer class is the class used in the Model1 and Model2 classes and acts as the "memory" of the DQN.

This file also contains some functions for neural net construction (three to be exact)