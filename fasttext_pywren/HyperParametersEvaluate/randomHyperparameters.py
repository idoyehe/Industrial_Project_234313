from random import choice
import numpy as np

list_size = 50

lr_values = np.arange(0.1, 1.1, 0.1)

lrUpdateRate_values = np.arange(20, 110, 10)

ws_values = np.arange(3, 8, 1)

epoch_values = np.arange(5, 16, 1)

iter_hyperparameters = list()
for i in range(list_size):
    hyperparameters_set = {
        "lr": choice(lr_values),
        "lrUpdateRate": choice(lrUpdateRate_values),
        "ws": choice(ws_values),
        "epoch": choice(epoch_values)}
    iter_hyperparameters.append([hyperparameters_set])

print(iter_hyperparameters)
