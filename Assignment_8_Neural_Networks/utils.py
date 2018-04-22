import os
import numpy as np

class data_loader():
    '''Breaks the data into batches and provides an iterator over them.
    '''
    def __init__(self, features, labels, batch_size=256, shuffle=True):
        self.batch_size = batch_size
        self.left = 0
        self.right = batch_size
        self.len = features.shape[1]

        if shuffle:
            random_idx = np.random.permutation(self.len)
            self.features = features[:, random_idx]
            self.labels = labels[random_idx]
        else:
            self.features = features
            self.labels = labels

    def __iter__(self):
        while self.right > self.left:
            yield (self.features[:, self.left:self.right], self.labels[self.left:self.right])
            self.left = self.right
            self.right += self.batch_size
            if self.right > self.len:
                self.right = self.len
                

def accuracy(output, Y):
    return np.sum(np.argmax(output, 0) == Y) / output.shape[1]
    

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

