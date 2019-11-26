import random
import socket
import struct
import numpy as np


class NeurosmashEnvironment:
    def __init__(self, ip = "127.0.0.1", port = 13000):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ip     = ip
        self.port   = port
        self.client.connect((ip, port))
        
        self.IMG_SHAPE = (256, 256, 3)

    def init(self):
        self._send(0, 1)
        return self._receive()

    def step(self, action):
        self._send(action, 2)
        return self._receive()

    def _receive(self):
        data_size = 1 + 1 + np.prod(self.IMG_SHAPE)
        data = self.client.recv(data_size)
        info   = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, 196610)]
        return info, reward, state
    
    
    def _receive_safe(self):
        while True:
            try:
                return self._receive()
            except IndexError:
                pass

    def _send(self, action, transition):
        self.client.send(bytes([action, transition]))
        
        
    def disconnect(self):
        self.client.close()


    def init_env(self):
        while True:
            try:
                return self.init()
                
            except IndexError:
                pass
        

    def step_safe(self, action):
        while True:
            try:
                return self.step(action)
            except IndexError:
                pass

