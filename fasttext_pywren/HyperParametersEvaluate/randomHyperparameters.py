from random import choice

list_size = 50

lr_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

lrUpdateRate_values = [20, 30, 40, 50, 60, 70, 80, 90, 100]

ws_values = [3, 4, 5, 6, 7]

epoch_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

iter_hyperparameters = list()
for i in range(list_size):
    hyperparameters_set = {
        "lr": choice(lr_values),
        "lrUpdateRate": choice(lrUpdateRate_values),
        "ws": choice(ws_values),
        "epoch": choice(epoch_values)}
    iter_hyperparameters.append([hyperparameters_set])

print(iter_hyperparameters)
