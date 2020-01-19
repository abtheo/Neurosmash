# coding: utf-8

# ### **SOW-MKI49-2019-SEM1-V: NeurIPS**
# 
# # Project: Neurosmash
# # Group 14


"""Policies: Different reward schemas to encourage different behaviours"""

#Try to Win + Survive as long as possible
def LiveLongAndProsper(reward, done=None, i=None):
    if reward == 0:
        return 1
    return reward

#RESULT: Converges to a very efficient suicide machine
def MeeseeksAndDestroy(reward, done, i):
    if done and not reward == 10:
        return -10000
    if reward == 0:
        return -1 * i * 0.001
    if reward == 10:
        return 10000

def SoreLoser(reward, done, i=None):
    if done and not reward == 10:
        return -1000
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
    