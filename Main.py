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

import torch
import torch.nn
import torch.nn.functional as F

#Local File Imports
from Env import NeurosmashEnvironment
from Agents import NeurosmashAgent


#environment = NeurosmashEnvironment()
environment = NeurosmashEnvironment(timescale=3)
agent = NeurosmashAgent()
#agent.load_state_dict(torch.load("D:/AI/Neurosmash/Brains/model_3.pt", map_location=torch.device('cpu')))


NUM_STEPS = 1000
rewards = []

game_memory = []
experience_replay = []

info, reward, state = environment.reset()
for _ in range(NUM_STEPS):
    #state_norm = [s/255 for s in state]
    #action = agent.step(state_norm)

    action = random.randint(0,2)
    #values, indices = torch.max(action, 1)
    #print(action)
   # print("Values", values, "Indices", indices)
    
    game_over, reward, state = environment.step(action)

    #Modify environment reward
    if reward == 0:
        reward = -1
    elif reward == 10:
        reward = 1000

    rewards.append(reward)
    
    game_memory.append([state, action])
    
    if game_over:
        print(sum(rewards))
        experience_replay.append(game_memory)   

        #Reset
        game_memory = []
        rewards = []
        info, reward, state = environment.reset()
        
# plt.plot(rewards)
# plt.show()


#Fitting CNN model:
def train_loop(train_agent, experience_replay, N_epochs=1): 
    #Loss Function
    loss_func = torch.nn.CrossEntropyLoss()
    #Optimizer
    optimizer = torch.optim.Adam(train_agent.parameters(), lr=0.0001)

    for epoch in range(N_epochs):
        for i,game in enumerate(experience_replay):
            #Reset loss counter
            running_loss = 0.0
            for j,steps in enumerate(game):
                state, action = steps[0], steps[1]

                state_norm = [s / 255 for s in state]            
                state_tensor = torch.tensor(state_norm, dtype=torch.float).view(3, 256, 256).view(1, 3, 256, 256).cuda()

                action_tensor = torch.tensor(action, dtype=torch.long).view(1).cuda()
                #print("Actual: ", action_tensor)
                        
                optimizer.zero_grad()
                outputs = train_agent(state_tensor)
                #print("Predicted: ", outputs)

                loss = loss_func(outputs, action_tensor)
                
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                #if (i+1) % 10 == 0:    # print every 10 steps
            print('[%d, %5d, %5d] loss: %.3f' %
            (epoch + 1,  i + 1,j+1,  running_loss / j ))
            running_loss = 0.0