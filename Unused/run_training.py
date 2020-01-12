
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
from Agents import NeurosmashAgent


#Init Agent
train_agent = NeurosmashAgent()   
print(train_agent)
train_agent.training = True
#Enable GPU if available
if torch.cuda.is_available():
  train_agent.cuda()

#Training CNN
N_epochs = 100

#Fitting CNN model:
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

#Save trained agent's brain
torch.save(train_agent.state_dict(), "./Brains/model.pt")