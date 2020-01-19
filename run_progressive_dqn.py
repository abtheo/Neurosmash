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
import traceback

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

#Def Environment
env = NeurosmashEnvironment(size=256, timescale=10)

#Hyperparams
n_episodes = 250
transfer_every = 5
batch_size = 4

# in_units = env.size * env.size * 3 # get dim of state space for input to Qnet
# out_units = 3 # get number of actions for Qnet output
# hidden_units = 128

#Def Agent's brain
#policy_net = QNetMLP(in_units, hidden_units, out_units)
#target_net = QNetMLP(in_units, hidden_units, out_units)
policy_net = NeurosmashAgent()
target_net = NeurosmashAgent()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#Init empty Memorys
memory = ReplayMemory(max_size=1024)
victory_memory = ReplayMemory(max_size=1024)
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
  torch.cuda.empty_cache()
#Catch KeyboardInterrupts and save model
#i=-1
# try:
#     #Reinforcement Loop
#     #for i in tqdm.trange(n_episodes):
#     while True:
#        i += 1
for i in range(n_episodes):
    info, reward, state = env.reset() # reset env before starting a new episode
    j=0
    shortterm_memory = ReplayMemory(max_size=256)
    while True:
        j += 1
        # interact with env
        action = agent.step(state)

        #observation, reward, done, info = env.step(action)
        done, base_reward, observation = env.step(action)

        #Determine real reward based on Policy
        reward = Policies.LiveLongAndProsper(base_reward, done)
        #reward = base_reward
        # store transaction in memory
        transition = [state, action, reward, observation, done]
        shortterm_memory.store(*transition)
        memory.store(*transition)

        # Step to next state
        state = observation
        
        # sample from memory and train policy
        if len(memory.buffer) > batch_size:
            if len(victory_memory.buffer) > batch_size:
                weighted_coin = random.uniform(0, 1)
                #coin_flip = bool(random.getrandbits(1))
                #if coin_flip:
                if weighted_coin > 0.1:
                    #Sample from winning games
                    train_batch = victory_memory.sample(batch_size)
                else:
                    train_batch = memory.sample(batch_size)
            else:
                #Random sample half the time to maintain learning from erroneous events
                train_batch = memory.sample(batch_size)
   
            loss = agent.train_policy(train_batch)
            
        # transfer weights from policynet to targetnet
        if j % transfer_every == 1:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        R[i] = base_reward
        #R += reward

        #Reset if game lasts too long:
        #Protects against environment bug where agents can be trapped outside the arena
        if j > 1000:
            break

        if done:
            #If agent wins, append the last N transitions to the victory memory
            if reward > 9:
                last_n = shortterm_memory.get_last(8)
                for n in last_n:
                    victory_memory.store(*n)
            print(f"\nEpisode: {i} Reward: {R[i]}")
            break

"""Save model, save rewards, plot wins, plot moving average"""
name = "p_LLAP_2"
torch.save(agent.policy_net.state_dict(), f"Brains/policy_brain_{name}.pt")
np.save(f"Results/{name}_rewards.npy", R)


window_width = 5
cumsum_vec = np.cumsum(np.insert(R, 0, 0)) 
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
plt.plot(range(len(ma_vec)), ma_vec)
plt.show()
