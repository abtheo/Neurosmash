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
import collections

import torch
import torch.nn
import torch.nn.functional as F

#Local File Imports
from Env import NeurosmashEnvironment
from DQN import DQNAgent, QNetMLP, NeurosmashAgent
from ExperienceReplay import ReplayMemory, Transition
import Policies

#Clear memory
agent = None
del agent
torch.cuda.empty_cache()

#Def Environment
env = NeurosmashEnvironment(size=256, timescale=10)

#Hyperparams
n_episodes = 500
transfer_every = n_episodes // 100

batch_size = 1
in_units = env.size * env.size * 3 # get dim of state space for input to Qnet
out_units = 3 # get number of actions for Qnet output
hidden_units = 128


#Def Agent's brain
#policy_net = QNetMLP(in_units, hidden_units, out_units)
#target_net = QNetMLP(in_units, hidden_units, out_units)
policy_net = NeurosmashAgent()
target_net = NeurosmashAgent()

agent.target_net.load_state_dict("Brains/target_brain.pt")
agent.policy_net.load_state_dict("Brains/policy_brain.pt")

#Def Memory
memory = ReplayMemory(max_size=1024) 

#Init lists
R = np.zeros(n_episodes)
reward = 0
losses = []
epses = []

#Init DQN agent
agent = DQNAgent(target_net, policy_net, memory)

if torch.cuda.is_available():
  print("Running on GPU")
  agent.target_net.cuda()
  agent.policy_net.cuda()
i=-1

#Reinforcement Loop
#for i in tqdm.trange(n_episodes):
while True:
    i = i+1
    info, reward, state = env.reset() # reset env before starting a new episode
    j=0
    R = 0
    while True:
        j += 1
        # interact with env
        action = agent.step(state)

        #observation, reward, done, info = env.step(action)
        done, reward, observation = env.step(action)

        #Determine real reward based on Policy
        #reward = Policies.SoreLoser(reward, done)

        # store transaction in memory
        memory.store(state, action, reward, observation, done)

        # Step to next state
        state = observation
        
        # sample from memory and train policy
        if len(memory.buffer) > batch_size:
            train_batch = memory.sample(batch_size)
        
            loss = agent.train_policy(train_batch)
            
        # transfer weights from policynet to targetnet
        if i % transfer_every == 1:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        #losses.append(loss)
        #epses.append(agent.eps)
        
        #R[i] += reward
        R += reward

        #Reset if game lasts too long:
        #Protects against environment bug where agents can be trapped outside the arena
        if j > 5000:
            break


        if done:
            print("\nReward" + str(i) + ": " + str(R))
            # if (i+1) % 10 == 0:
            #     avg = sum(R[i-10:i]) / 10
            #     print("Average over last 10 games: ", avg)
            break
