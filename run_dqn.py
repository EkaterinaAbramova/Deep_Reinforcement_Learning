#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document : TUTORIAL : DQN using Deep Reinforcement Learning (with GPUs).
           Cart Pole: deterministic, episodic.
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

#%% Setup
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque # pronounced "deck" : double-ended queue
from itertools import count

import torch
import torch.nn as nn # neural network
import torch.optim as optim  # optimization 
import torch.nn.functional as F # convolution functions

# deterministic environment
env = gym.make("CartPole-v1")
print(env.spec.max_episode_steps) # episode ends automatically after these many steps (500)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "mps" if torch.backends.mps.is_available() else "cpu"
# TEST next line if output produced, if yes GPUs worked well
# x = torch.rand(size=(3, 4)).to(device)


#%% Memory Replay 
'''
We’ll be using experience replay memory for training our DQN. It stores the 
transitions that the agent observes, allowing us to reuse this data later. 
By sampling from it randomly, the transitions that build up a batch are 
decorrelated. This breaks the correlation between sequential experiences, 
making the updates more stable. It has been shown that this greatly stabilizes 
and improves the DQN training procedure.
'''

# single transition in our environment
'''
namedtuple - allows me to define simple classes for creating tuple objects with 
names to make code more readable.

Point = namedtuple('Point', ['x', 'y'])
p = Point(11, 22)
print(p.x)  # Outputs: 11
'''
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


# ReplayMemory class is a Cyclic buffer of bounded size that holds the transitions observed recently
'''
Buffer is a data structure that serves as a holding area, allowing computations 
to continue without waiting for slower I/O operations to complete. This can 
help in optimizing performance because faster processes don't have to wait for 
slower processes to catch up.

Cyclic buffer uses a single, fixed-size buffer as if it were connected end-to-end 
(forming a circle). When it fills up, adding another element overwrites the 
oldest element, making it "cyclic."

Deque : generalization of stacks & queues which supports thread-safe, memory-efficient 
appends and pops from either side of the deque.
Deque is used for the ReplayMemory to efficiently handle the addition and 
removal of experiences

Namedtuple, Transition, is used to structure and access individual experiences in a human readable manner.

memory.push(state, action, next_state, reward) # ensure to match the order  
The *args would collect these individual arguments into a tuple, and then 
Transition(*args) would unpack them to create a Transition namedtuple.
'''
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) # initialised to 10000 below

    def push(self, *args): 
        # save a transition (*args is used to pass a variable number of non-keyworded arguments). Makes code more concise 
        self.memory.append(Transition(*args)) # Transition is defined as ('state', 'action', 'next_state', 'reward')
    
    #  selecting a random batch of transitions (batch_size) from a given sequence (self.memory), for training
    def sample(self, batch_size): 
        return random.sample(self.memory, batch_size)

    # get number of transitions stored in memory
    def __len__(self):
        return len(self.memory)


#%% DQN Algorithm
'''
Uses Q function, td-error and minimises this error uisng Huber loss. 

Our model will be a feed forward neural network that takes in the difference 
between the current and previous screen patches.

It has two outputs Q(s,left) and Q(s,right) (where s is the input to network). 
Network is trying to predict the expected return of taking each action given current input.

nn.Module :
    - is a base class for all neural network modules in PyTorch 
    - provides methods to retrieve all parameters (weights; biases; gradients for all these parameters etc). 
    - provides utility functions, such as 
        to(device) to move your model to a GPU, and methods like 
        state_dict() and load_state_dict() to save and load the model's parameters.

Within a custom module (a class that inherits from nn.Module), you usually define layers in the constructor. 

For custom module to be callable and to perform its computation, you need to define a forward method. 
'''

# DQN class, which inherits from nn.Module. This means that DQN will be a type 
# of neural network module, with all the capabilities provided by PyTorch's nn.Module.
class DQN(nn.Module):
    
    # n_observations :  length of state (4, i.e. cart position, cart velocity, pole angle, pole angular velocity)
    # n_actions : number of actions (2 i.e. left, right)
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__() #  calls the initializer of the parent class (nn.Module)
        # 3 linear (fully connected) layers of the neural network
        self.layer1 = nn.Linear(n_observations, 128) # takes n_observations inputs and maps them to 128 outputs.
        self.layer2 = nn.Linear(128, 128) # takes 128 outputs from previous layer and maps them to another 128 outputs
        self.layer3 = nn.Linear(128, n_actions) # takes 128 outputs from 2nd layer and maps them to n_actions outputs
        
    # FORWARD PASS
    # Called with either one element to determine next action, or a batch during optimization. 
    # Returns tensor([[left0exp,right0exp]...])
    def forward(self, x): # input tensor x
        # ReLU activation function
        x = F.relu(self.layer1(x)) # hidden layer 1
        x = F.relu(self.layer2(x)) # hidden layer 2
        # output layer : Q-value prediction for each action
        return self.layer3(x) # want the raw Q-values, so the output is not passed through an activation function


#%% Training : Hyperparameters and utilities
'''
The probability of choosing a random action will start at EPS_START and will 
decay exponentially towards EPS_END. EPS_DECAY controls the rate of the decay.
'''

BATCH_SIZE = 128 # number of transitions sampled from the replay buffer
GAMMA = 0.99 # discount factor (number closer to 1 means past is disregarded more)
EPS_START = 0.9 # starting probability of choosing a random action
EPS_END = 0.05 # final value of epsilon
EPS_DECAY = 1000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 1e-4 # learning rate of the ``AdamW`` optimizer

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# policy network : neural network model that approximates the Q-values for a given state.
# i.e.used to decide what action to take given current state (or observation) in DQN.
# its output is a tensor where each entry corresponds to the predicted Q-value of a particular action for that state.
policy_net = DQN(n_observations, n_actions).to(device)

# target network : used to compute target Q-values for updating the policy network. Helps stabilize the learning process in DDQN.
target_net = DQN(n_observations, n_actions).to(device)
# copy weights and biases (collectively referred to as "state_dict") from the policy_net to the target_net. Initially, both networks start with the same parameters.
target_net.load_state_dict(policy_net.state_dict())

'''
AdamW is a variant of the Adam optimizer which decouples the weight decay from 
the optimization steps. It's known to provide better regularization and has 
become a popular choice in training deep neural networks.

policy_net.parameters() parameters the optimizer will update during training.
lr=LR learning rate

AMSGrad : variant of Adam optimizer that improve its convergence properties by
maintaining the maximum of past squared gradients in the denominator.

By defining this optimizer, you are preparing for the training loop, where, 
after calculating the gradients using backward(), you'll use the optimizer to 
adjust the weights and biases of the policy_net in the direction that minimizes the loss.
'''
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Create an object 'memory' from ReplayMemory class that will hold a max of 10000 recent transitions. 
# Once it reaches 10,000, the oldest experiences will be overwritten by the newest ones: its a cyclic buffer.
memory = ReplayMemory(10000)

# select an action according to an epsilon greedy policy
steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random() # uniform random number [0.0, 1.0) inclusive of 0, exclusive of 1
    # calculate current ε (eps_threshold), as the number of steps increases, ε decreases.
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # get greedy action
    if sample > eps_threshold: 
        '''
        with torch.no_grad(): context manager in PyTorch temporarily disables 
        gradient computation for the operations enclosed within its block. 
        This means that operations inside this block will not be tracked for gradients.
        '''
        with torch.no_grad(): # only need forward pass evaluation, so used for memory efficiency
            '''
            q_values: 
            Dimension 0: Batch dimension
            If you forward propagate a single state through the network, the size of this dimension will be 1. 
            However, when you train neural networks, you often send multiple data samples at once (a batch) for efficiency. In such cases, the size of this dimension corresponds to the number of samples in the batch.
            Dimension 1: Action dimension
            CartPole, there are 2 actions (move left or move right). So the size of this dimension will be 2, and each value in this dimension represents the Q-value for a specific action.
            '''
            # Forward the state through the policy network to get Q-values
            q_values = policy_net(state)
            # Get the maximum Q-value along dimension 1 (action dimension) and get the index of the max value
            max_q_value, max_index = q_values.max(1)
            # Reshape the index tensor into a 2D tensor with one row and one column
            greedy_action = max_index.view(1, 1) # index / position of the action is the action number
            return greedy_action
    # select action at random
    else: 
        # Sample a random action from the environment's action space (any action can be selected i.e. greedy or any non greedy)
        sampled_action = env.action_space.sample()
        # Place the sampled action inside a 2D list
        action_as_list = [[sampled_action]]
        # Convert the 2D list into a PyTorch tensor with the specified device and data type
        random_action = torch.tensor(action_as_list, device=device, dtype=torch.long)
        return random_action


# plotting duration of episodes (i.e. for how many steps did agent manage to hold the pole upright), along with an average over the last 100 episodes (orange line)
episode_durations = []
def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    #
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


#%% Optimise Model : single step of optimisation
'''
Key step in training deep Q-networks (DQNs) using the Q-learning algorithm, 
which involves updating the policy network based on sampled transitions from 
experience replay memory. 

Steps:
- sample a batch
- concatenate all the tensors into a single one
- compute Q(s_t,a_t) and V(s_t+1)
- combine them into the loss

set V(s)=0 if s is a terminal state
use target network to compute V(s_t+1) for added stability

The target network is updated at every step with a soft update controlled by 
the hyperparameter TAU.
'''

# perform a single step of the optimization
def optimize_model():
    # if not enough transitions in memory exit without updating the model
    if len(memory) < BATCH_SIZE:
        return
    
    # sample a batch of transitions from replay buffer
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch 
    batch = Transition(*zip(*transitions)) # convert batch-array of Transitions to Transition of batch-arrays (see https://stackoverflow.com/a/19343/3343043)

    '''
    Mask : generally refers to a way of selecting specific elements from data, for the purpose of processing them
    final state : would've been the one after which simulation ended
    '''
    # BOOLEAN TENSOR:
    # non_final_mask is a boolean tensor of same length as batch.next_state 
    # it is a mask indicating which entries in batch.next_state are non-terminal 
    # it has True at positions where the corresponding state is not a terminal, and False otherwise
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    
    # NON FINAL STATES TENSOR:
    # filter out None values from batch.next_state and concatenate resulting list of non-terminal states into a single tensor
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    # State, action, and reward tensors are concatenated to form batches.
    state_batch  = torch.cat(batch.state) # batch of states, each of which we want to predict Q-values for.
    action_batch = torch.cat(batch.action) # 2D tensor containing indices of actions that were actually taken in each state. Each index refers to a column in the output of policy_net(state_batch). 
    reward_batch = torch.cat(batch.reward)
    
    '''
    Use policy network to compute Q-values for all actions in the current states (state_batch)
    Among these Q-values, get the Q-values corresponding to the taken actions (action_batch) are selected.
    '''
    # get the Q-values for all actions for each state in the batch
    q_values_all_actions = policy_net(state_batch) # produces 2D tensor of shape (batch_size, number_of_actions). Each row corresponds to the Q-values predicted for each action of a single state.
    # for each state, select the Q-value of the action that was taken 
    state_action_values = q_values_all_actions.gather(1, action_batch) # gather is indexing for tensors. 1 is dimension along which to gather.

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    
    '''
    Next computing elements of this key equation, however not quite directly as using neural network.
    Q(s_t, a_t) ← Q(s_t, a_t) + α [ r_t + γ*max_a Q(s_t+1,a) - Q(s_t, a_t) ]
    '''
    # Compute the expected Q values; this is the TD target: r_t + γ*max_a Q(s_t+1,a)
    expected_state_action_values = reward_batch + (GAMMA * next_state_values) 

    # Compute Huber Loss (combination of MSE and MAE; makes loss less sensitive to outliers)
    '''
    for small differences between predicted and target values, it calculates MSE
    for larger differences, it calculates MAE
    '''
    criterion = nn.SmoothL1Loss()
    # The loss is akin to the TD error, though it's not a direct equivalent : r_t + γ*max_a Q(s_t+1,a) - Q(s_t, a_t)
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    '''
    Clear any old gradients.
    Compute new gradients based on the current state of the model and the loss.
    Clip the gradients to ensure stable training.
    Update the model's parameters using the optimizer and the gradients.
    '''
    # Optimize the model
    optimizer.zero_grad() # before computing the backward pass (calculating the gradients), it's essential to zero out any existing gradients. As gradients are accumulated in PyTorch by default. If you don't zero them out, you'd accumulate gradients across multiple forward and backward passes, which is not what we want.
    loss.backward() # computes the gradient of the loss with respect to model parameters 
    
    # In-place gradient clipping (cap the gradients to prevent them from getting too large and thus skipping over minima when taking a large step)
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100) # gradients are capped at value abs(100)
    # adjust each parameter based on the gradients computed during loss.backward()
    optimizer.step() # The learning rate α of the key Q update equation is abstracted away in this optimizer's update rule


#%% Training Loop : code for training our model
'''
Start: reset the environment and obtain the initial state Tensor
sample an action
execute it
observe the next state and the reward (always 1)
optimize the model once. 
When the episode ends (our model fails or it reaches 500 episode steps), we restart the loop.

Below, num_episodes is set to 500 if a GPU is available, otherwise use 50 so 
training does not take too long. However, 50 episodes is insufficient to observe 
good performance on CartPole. You should see the model constantly achieve 500 
steps within 600 training episodes. Training RL agents can be a noisy process, 
so restarting training can produce better results if convergence is not observed.
'''
num_episodes = 500

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    # itertools.count() is an infinite iterator. It returns a generator that produces consecutive integers until you break. By default, it starts at 0 and increments by 1.
    for t in count():
        action = select_action(state) # select action based on epsilon greedy policy
        # action.item() :  method used to get the value of action tensor as a standard Python number. 
        # terminated : boolean
        # truncated : boolean indicating if episode was truncated before its natural termination condition (e.g. maximum step count)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ * θ + (1 −τ) * θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()

