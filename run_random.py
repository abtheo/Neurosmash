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
import torch
import torch.nn
import pickle 
import glob

#Local File Imports
from Env import NeurosmashEnvironment

NUM_STEPS = 10000
rewards = []
game_memory = []
experience_replay = []

#Initialise environment
environment = NeurosmashEnvironment()
environment.init_env()
for _ in range(NUM_STEPS):
    #action = agent.step(info, reward, state)
    action = random.randint(0,2)
    
    game_over, reward, state = environment.step_safe(action)
    
    rewards.append(reward)
    
    game_memory.append([state, action])
    
    if game_over:
        print(sum(rewards))
        #If Red wins:
        if 100 in rewards:
            experience_replay.append(game_memory)    
        #Reset
        game_memory = []
        rewards = []
        environment.init_env()
        

#Plot reward history      
# plt.plot(rewards)
# plt.show()

#Pickle experience replay for training data
filepaths = glob.glob("./Replays/")
index = str(len(filepaths))
with open("./Replays/randomwalk_" + index + ".pkl", 'wb') as f:
    pickle.dump(experience_replay, f)


