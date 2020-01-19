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

#Def Agent's brain
policy_net = NeurosmashAgent()
target_net = NeurosmashAgent()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

#Init empty Memory
memory = ReplayMemory(max_size=1024)
#Init lists
reward = 0
i=-1
losses = []
epses = []
R = np.zeros(n_episodes)

#Init DQN agent
agent = DQNAgent(target_net, policy_net, memory)

if torch.cuda.is_available():
  print("Running on GPU")
  agent.target_net.cuda()
  agent.policy_net.cuda()
  torch.cuda.empty_cache()
#Catch KeyboardInterrupts and save model
for i in range(n_episodes):
    #i += 1
    info, reward, state = env.reset() # reset env before starting a new episode
    j=0
    #R = 0
    while True:
        j += 1
        # interact with env
        action = agent.step(state)

        #observation, reward, done, info = env.step(action)
        done, base_reward, observation = env.step(action)

        #Determine real reward based on Policy
        reward = Policies.SoreLoser(base_reward, done)
        #reward = base_reward

        # store transaction in memory
        transition = [state, action, reward, observation, done]
        memory.store(*transition)

        # Step to next state
        state = observation
        
        # sample from memory and train policy
        if len(memory.buffer) > batch_size:
            train_batch = memory.sample(batch_size)       
            loss = agent.train_policy(train_batch)
            
        # transfer weights from policynet to targetnet
        if j % transfer_every == 1:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        #Track original reward for comparisons
        R[i] = base_reward
        #R += reward

        #Reset if game lasts too long:
        #Protects against environment bug where agents can be trapped outside the arena
        if j > 1000:
            print("Game too long, break")
            break

        if done:
            print(f"\nEpisode: {i} Reward: {R[i]}")
            break

"""Save model, save rewards, plot wins, plot moving average"""
name = "v_soreloser_2"
torch.save(agent.policy_net.state_dict(), f"Brains/policy_brain_{name}.pt")
np.save(f"Results/{name}_rewards.npy", R)
print("Save complete")

#Plot Moving Average
window_width = 10
cumsum_vec = np.cumsum(np.insert(R, 0, 0)) 
ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
plt.plot(range(len(ma_vec)), ma_vec)
plt.show()


# except Exception as e:
#     print(e)
#     print(traceback.format_exc())
#     #Save on intentional keyboard exit
#     if type(e) == KeyboardInterrupt:
#         print("Saving...")
#         torch.save(agent.target_net.state_dict(), "Brains/target_brain.pt")
#         torch.save(agent.policy_net.state_dict(), "Brains/policy_brain.pt")
#         print("Saved!")


