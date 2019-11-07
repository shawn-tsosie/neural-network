import glob
import os
import sys
import torch

from pathlib import Path

from neural_network import *


class Data(object):

    def __init__(self, file_path, batch_size=1, shuffle=True):
        self.path = Path(file_path)
        self.batch_size = batch_size
        self.shuffle = True
        self.data = None
        self.train_data = None
        self.test_data = None
        self.val_data = None

    def get_data(self):
        pass

    def initialize_data(self):
        pass


if __name__ == '__main__':
    print('Beginning testing')
    test = layers.Layer(5, 4, 'hi')
    print(test.weights)
