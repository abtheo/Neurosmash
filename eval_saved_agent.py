# coding: utf-8

# ### **SOW-MKI49-2019-SEM1-V: NeurIPS**
# 
# # Project: Neurosmash
# # Group 14

import random
import socket
import struct
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
import matplotlib.pyplot as plt
import collections
import sys

import torch
import torch.nn
import torch.nn.functional as F

#Local File Imports
from Env import NeurosmashEnvironment
from DQN import DQNAgent, QNetMLP, NeurosmashAgent
from ExperienceReplay import ReplayMemory, Transition
import Policies

#CLI params


#Clear memory
agent = None
del agent

#Def Environment
env = NeurosmashEnvironment(size=256, timescale=10)

#Hyperparams
n_episodes = 25
transfer_every = n_episodes // 100

batch_size = 1

#Def Agent's brain
policy_net = NeurosmashAgent()
target_net = NeurosmashAgent()

#Def Memory
memory = ReplayMemory(max_size=1024) 

#Init lists
R = np.zeros(n_episodes)
reward = 0
losses = []
epses = []

#Init DQN agent
agent = DQNAgent(target_net, policy_net, memory)

#agent.target_net.load_state_dict(torch.load("Brains/target_brain_b4.pt"))
agent.policy_net.load_state_dict(torch.load("Brains/policy_brain_b4.pt"))
agent.target_net.load_state_dict(agent.policy_net.state_dict())

if torch.cuda.is_available():
  torch.cuda.empty_cache()
  print("Running on GPU")
  agent.target_net.cuda()
  agent.policy_net.cuda()

#Reinforcement Loop
for i in range(n_episodes):
    info, reward, state = env.reset() # reset env before starting a new episode
    j=0
    while True:
        j += 1
        # interact with env
        action = agent.step(state, decay_enabled=False)

        #observation, reward, done, info = env.step(action)
        done, reward, observation = env.step(action)

        #Determine real reward based on Policy
        #reward = Policies.SoreLoser(reward, done)

        # Step to next state
        state = observation
        
        #Save rewards for evaluation
        R[i] = reward

        #Reset if game lasts too long:
        #Protects against environment bug where agents can be trapped outside the arena
        if j > 1000:
            break

        if done:
            print("\nReward" + str(i) + ": " + str(R[i]))
            # if (i+1) % 10 == 0:
            #     avg = sum(R[i-10:i]) / 10
            #     print("Average over last 10 games: ", avg)
            break


plt.scatter(range(n_episodes), R)
plt.show()