import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQNAgent:
    
    # constants for the exploration decay
    eps_start = 1.0
    eps_min = 0.02
    eps_decay = 5000
    
    
    def __init__(self, target_net, policy_net, memory):
        
        # target and policy networks with the respective optimizers
        self.target_net = target_net
        self.policy_net = policy_net
        self.optim_policy = optim.RMSprop(self.policy_net.parameters())
        self.optim_target = optim.RMSprop(self.target_net.parameters())
        
        self.num_actions = self.policy_net.out_units
        
        self.discount_factor = 0.98
        
        # hyperparameters
        self.eps = DQNAgent.eps_start # exploration probability
        self.steps_done = -1 # because decay will be triggered before action is done
        
    def step(self, state):
        """
        Based on the current state of the environment, expoloration coefficient 
        and the policy network select an action 
        """
        # decay eps 
        self.steps_done += 1
        self.exploration_decay()
        
        # pick action
        if np.random.rand() < self.eps: 
            return np.random.randint(self.num_actions)
        else:
            with torch.no_grad():
                # get actions value from policy net and return the index of max action
                return self.policy_net(torch.Tensor(state).cuda()).argmax().item()
                                  

                                  
    def calculate_target(self, train_batch):
        """
        Calculate the target value of a batch of training data
        
        if done: y = r
        otherwise: y = r + discount_factor * Q_max(s`)
        """
                                  
        y = np.array([step.r for step in train_batch]) # init y = r
        
        not_dones = np.array([not step.done for step in train_batch])                        
                                 
        # get Q_max for future state of every transition
        next_states = torch.Tensor([step.s1 for step in train_batch]).cuda()
        with torch.no_grad():
            Q = self.target_net(next_states).cpu().numpy().max(axis=1)
        
        # add the discounted Q value only if not done
        y[not_dones] = self.discount_factor * Q[not_dones]
        
        return y
    
 
    
    
    def train_policy(self, train_batch):
        """
        train policy network the target value
        """
        self.optim_policy.zero_grad()
        se = nn.MSELoss(reduction="sum") # define sum squared error
        # calculate target
        y = torch.Tensor(self.calculate_target(train_batch)).cuda()
        # collect states and compute Qmax
        states = torch.Tensor([step.s0 for step in train_batch]).cuda()
        Q, _ = torch.max(self.policy_net(states), 1)
       
        # train policy net
        loss = se(Q,y)
        loss.backward()
        self.optim_policy.step()
            
        return loss.item() # return the value of loss for tracking the progress
            
            
    def exploration_decay(self):
        self.eps = DQNAgent.eps_min + (DQNAgent.eps_start - DQNAgent.eps_min) * np.exp(-self.steps_done / DQNAgent.eps_decay)
        
    
        
class QNetMLP(nn.Module):
    
    def __init__(self, in_units, hidden_units, out_units):
        super(QNetMLP, self).__init__()
        self.out_units = out_units
        self.l1 = nn.Linear(in_units, hidden_units)
        self.l2 = nn.Linear(hidden_units, out_units)

        
    def forward(self, x):
        #Convert to GPU format
        x.cuda()
        return self.l2(F.relu(self.l1(x)))
        