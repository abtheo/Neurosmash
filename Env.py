import random
import socket
import struct
import numpy as np
from PIL import Image


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
        # Kudos to Jan for the socket.MSG_WAITALL fix!
        data   = self.client.recv(2 + 3 * self.size ** 2, socket.MSG_WAITALL)
        end    = data[0]
        reward = data[1]
        state  = [data[i] for i in range(2, len(data))]

        return end, reward, state

    def state2image(self, state):
        return Image.fromarray(np.array(state, "uint8").reshape(self.size, self.size, 3))

    
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

