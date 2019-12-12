# coding: utf-8

# ### **SOW-MKI49-2019-SEM1-V: NeurIPS**
# 
# # Project: Neurosmash
# # Group 13


"""Policies: Different reward schemas to encourage different behaviours"""

#Try to Win + Survive as long as possible
def LiveLongAndProsper(reward):
    if reward == 0:
        return 1
    elif reward == 10:
        return 1000

#Try to win as quickly as possible.
#To avoid reward hacking, strongly penalise death
def SeekAndDestroy(reward, done):
    if done and not reward == 10:
        return -1000
    if reward == 0:
        return -1
    if reward == 10:
        return 10000

#RESULT: Converges to a very efficient suicide machine
def MeeseeksAndDestroy(reward, done, i):
    if done and not reward == 10:
        return -10000
    if reward == 0:
        return -1 * i * 0.01
    if reward == 10:
        return 10000

def SoreLoser(reward, done):
    if done and not reward == 10:
        return -100
    if reward == 10:
        return 1000
    else:
        return reward

def DeathAndDecay(reward, done, i):
    #Win
    if done and reward == 10:
        print("Win", i)
        return 10000 - (i*2)
    #Lose
    elif done and not reward == 10:
        print("Lose", i)
        return -1000
    else:
        return 1
    