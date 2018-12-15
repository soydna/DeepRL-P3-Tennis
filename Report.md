## Introduction
This implementation is heavily based on my implementation for project 2 (https://github.com/soydna/Deep-RL-P2). The only change is 
that the agent accepts a pair of states and returns a pair of actions. The replay buffer is shared and there is only one actor and critc.
Identical hyperparameters as in P2.

## Architecture
Based on Deep Deterministic Policy Gradients (DDPG) (https://arxiv.org/pdf/1509.02971.pdf)  
Both Actor and Critic uses a fully connected networks with two hidden layers:
* Actor has 150 and 100 nodes in the hidden layers
* Critic har 150 (+ 2 actions) in the first hidden layer and 150 nodes in the second hidden layer

Leaky Relu Activation to avoid dead relu problem with normal Relu.  
Batch normalization is used, to the L2 Weigth decay is set to zero (https://arxiv.org/abs/1706.05350)
Also the network is updated every 20 time steps, but for 10 iterations. This allows for more experience gathering between each learning updates which helps stablize training.

## Hyperparameters
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4       # learning rate of the critic
WEIGHT_DECAY = 0.0   # L2 weight decay


## Episodes to reach moving avreage (100) of target of 0.5
6440


## Future Work
* Try other hyperparameters
* Try other RL aproaches as A2C and D4PG
* 
