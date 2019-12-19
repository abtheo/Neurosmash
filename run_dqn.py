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
torch.cuda.empty_cache()

#Def Environment
env = NeurosmashEnvironment(size=256, timescale=1)

#Hyperparams
n_episodes = 500
transfer_every = 10
max_batch_size = 16
batch_size = 4

in_units = env.size * env.size * 3 # get dim of state space for input to Qnet
out_units = 3 # get number of actions for Qnet output
hidden_units = 128


#Def Agent's brain
#policy_net = QNetMLP(in_units, hidden_units, out_units)
#target_net = QNetMLP(in_units, hidden_units, out_units)
policy_net = NeurosmashAgent()
target_net = NeurosmashAgent()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#Init empty Memorys
memory = ReplayMemory(max_size=1024)
victory_memory = ReplayMemory(max_size=1024, max_batch_size=max_batch_size)
#Init lists
R = np.zeros(n_episodes)
reward = 0
i=-1
losses = []
epses = []

#Init DQN agent
agent = DQNAgent(target_net, policy_net, memory)

if torch.cuda.is_available():
  print("Running on GPU")
  agent.target_net.cuda()
  agent.policy_net.cuda()
#Catch KeyboardInterrupts and save model
try:
    #Reinforcement Loop
    #for i in tqdm.trange(n_episodes):
    while True:
        i += 1
        info, reward, state = env.reset() # reset env before starting a new episode
        j=0
        R = 0
        shortterm_memory = ReplayMemory(max_size=256)
        while True:
            j += 1
            # interact with env
            action = agent.step(state)

            #observation, reward, done, info = env.step(action)
            done, reward, observation = env.step(action)

            #Determine real reward based on Policy
            reward = Policies.SoreLoser(reward, done)

            # store transaction in memory
            transition = [state, action, reward, observation, done]
            shortterm_memory.store(*transition)
            memory.store(*transition)

            # Step to next state
            state = observation
            
            # sample from memory and train policy
            if len(memory.buffer) > batch_size:
                if len(victory_memory.buffer) > batch_size:
                    coin_flip = bool(random.getrandbits(1))
                    if coin_flip:
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
            
            #R[i] += reward
            R += reward

            #Reset if game lasts too long:
            #Protects against environment bug where agents can be trapped outside the arena
            if j > 5000:
                break

            if done:
                #If agent wins, append the last N transitions to the victory memory
                if reward > 9:
                    last_n = shortterm_memory.get_last(16)
                    for n in last_n:
                        victory_memory.store(*n)
                print(f"\nEpisode: {i} Reward: {R}")
                # if (i) % 10 == 0:
                #     avg = sum(R[i-10:i]) / 10
                #     print("Average reward over last 10 games: ", avg)
                break
except Exception as e:
    print(e)
    print(traceback.format_exc())
    #Save on intentional keyboard exit
    if type(e) == KeyboardInterrupt:
        torch.save(agent.target_net.state_dict(), "Brains/target_brain.pt")
        torch.save(agent.policy_net.state_dict(), "Brains/policy_brain.pt")
