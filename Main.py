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
agent = NeurosmashAgent()
agent.load_state_dict(torch.load("D:/AI/Neurosmash/Brains/model_3.pt", map_location=torch.device('cpu')))


NUM_STEPS = 1000
rewards = []

game_memory = []
experience_replay = []

info, reward, state = environment.init_env()
for _ in range(NUM_STEPS):
    state_norm = [s/255 for s in state]
    action = agent.step(state_norm)
    #action = random.randint(0,2)
    values, indices = torch.max(action, 1)
    print(action)
    print("Values", values, "Indices", indices)
    
    game_over, reward, state = environment.step_safe(indices)
    if reward == 0:
        game_over = True
    rewards.append(reward)
    
    game_memory.append([state, action])
    
    if game_over:
        print(sum(rewards))
        if 100 in rewards:
            experience_replay.append(game_memory)   
        
        game_memory = []
        rewards = []
        info, reward, state = environment.init_env()
        
plt.plot(rewards)
plt.show()



