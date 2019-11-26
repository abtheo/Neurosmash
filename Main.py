# coding: utf-8

# ### **SOW-MKI49-2019-SEM1-V: NeurIPS**
# 
# # Project: Neurosmash
# # Group 13

import random
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn
import torch.nn.functional as F

#Local File Imports
from Env import NeurosmashEnvironment
from Agents import NeurosmashAgent
    

environment = NeurosmashEnvironment()
NUM_STEPS = 1000
rewards = []

game_memory = []
experience_replay = []

info, reward, state = init_env(environment)
for _ in range(NUM_STEPS):
    action = train_agent.step(state)
    #action = random.randint(0,2)
    values, indices = torch.max(action, 1)
    print(action)
    print("Values", values, "Indices", indices)
    
    game_over, reward, state = step_safe(environment, indices)
    
    rewards.append(reward)
    
    game_memory.append([state, action])
    
    if game_over:
        print(sum(rewards))
        if 100 in rewards:
            experience_replay.append(game_memory)

        
        
        game_memory = []
        rewards = []
        init_env(environment)
        
        
plt.plot(rewards)
plt.show()


