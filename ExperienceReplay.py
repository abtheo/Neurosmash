import collections

class ExperienceReplay:

    def __init__(self, maxlen=10):
        self.max_length = maxlen
        self.replay = collections.deque(maxlen=maxlen)

    #Attempt to append to 
    def try_append(self, game):
        