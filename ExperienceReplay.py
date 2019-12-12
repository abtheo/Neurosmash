import collections
import random

Transition = collections.namedtuple("transition",
                        ("s0", "a", "r", "s1", "done"))

class ReplayMemory:
    """
    Store memory for the DQN by giving a transition
    sample to get a batch of training data
    """
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.pos = 0 # position of the next element
        
    def store(self, s0, a, r, s1, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None) # increase buffer to maximum capacity whenever nessesary
        self.buffer[self.pos] = Transition(s0, a, r, s1, done)
        self.pos = (self.pos + 1) % self.max_size
    
    def sample(self, batch_size):
        return random.sample(self.buffer, k=batch_size)