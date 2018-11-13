import pywren_ibm_cloud as pywren
from random import random
from time import time
from math import exp
from scipy.stats import norm
from numpy import median


class StockData:
    def __init__(self, name, drift, std_dev, last_value, days2predict):
        self.name = name
        self.days2predict = days2predict
        self.last_value = last_value
        self.std_dev = std_dev
        self.drift = drift

    def prediction(self):
        predicts_est = [self.last_value]
        for predict in range(1, self.days2predict + 1):
            rand = random()
            pow_r = norm.ppf(rand)
            predicts_est.append(predicts_est[predict - 1] * exp(self.drift + (self.std_dev * pow_r)))
        return predicts_est


ACTIONS = 2000
TOTAL = ACTIONS
print("Total Random Samples is: " + str(TOTAL))

gold = StockData(name="GOLD", drift=0.000142559, std_dev=0.010561899, last_value=1296.5, days2predict=219)
mlnx = StockData(name="Mellanox", drift=0.000581742829, std_dev=0.029879238, last_value=64.7, days2predict=219)
ibm = StockData(name="IBM", drift=0.000091967236, std_dev=0.012404562, last_value=153.42, days2predict=219)
nvda = StockData(name="Nvdia", drift=0.000936809, std_dev=0.027145343, last_value=193.5, days2predict=219)

stock = gold

iterdata = [[]] * ACTIONS


def my_map_function():
    return stock.prediction()


def my_reduce_function(list_of_lists):
    return [median(x) for x in zip(*list_of_lists)]


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""

# FLAG = "LOCAL"
FLAG = "CLOUD"
if FLAG == "LOCAL":
    start_time = time()
    for i in range(ACTIONS):
        iterdata[i] = my_map_function()
    print("Amortized Values are: ")
    print(my_reduce_function(iterdata))
else:
    start_time = time()
    pw = pywren.ibm_cf_executor()
    pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
    result_list = pw.get_result()
    print("Amortized Values are: ")
    print(result_list)
elapsed = time()

print("\nDuration: " + str(elapsed - start_time) + " Sec")
