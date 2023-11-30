from keras.utils import Sequence
from os import listdir, remove
from os.path import join
from math import ceil
import numpy as np
import Config
import datetime

class DataLoader(Sequence):
    def __init__(self, directory_x, directory_y, batch_size):
        files = listdir(directory_x)[0:24]
        self.x = [join(directory_x, file) for file in files]
        self.y = [join(directory_y, file) for file in files]
        self.max_index = len(files)
        self.indices=np.arange(self.max_index)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        return ceil(self.max_index / self.batch_size)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:min((index + 1) * self.batch_size,self.max_index)]
        result = np.stack([np.load(self.x[i])[:,:,0:3] for i in indices]).astype(np.float16), np.stack([np.load(self.y[i]) for i in indices]).astype(np.float16)
        return result
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


def check_folders(directory_x, directory_y, clean=False):
    filenames_x = listdir(directory_x)
    filenames_y = listdir(directory_y)
    for x in filenames_x:
        if not filenames_y.__contains__(x):
            print(f"{x} not found in the y folder. ")
            if clean:
                remove(join(directory_x,x))




