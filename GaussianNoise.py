import numpy as np


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class GaussianNoise:

    def __init__(self, action_dimension, mu=0, sigma=0.1):
        self.action_dimension = action_dimension
        self.mu = mu
        self.sigma = sigma

    def noise(self, batch_dim=1):
        return self.sigma * np.random.randn(batch_dim, self.action_dimension) + self.mu
