from random import choice
import numpy as np

lr_values = np.arange(0.1, 1.1, 0.1)
lrUpdateRate_values = np.arange(20, 110, 10)
ws_values = np.arange(3, 8, 1)
epoch_values = np.array([1, 5, 10])


def random_search(sets_num: int = 10):
    iter_hyperparameters = []
    for i in range(sets_num):
        hyperparameters_set = {
            "lr": choice(lr_values),
            "lrUpdateRate": choice(lrUpdateRate_values),
            "ws": choice(ws_values),
            "epoch": choice(epoch_values)}
        iter_hyperparameters.append([hyperparameters_set])
    return iter_hyperparameters
