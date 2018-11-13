import pywren_ibm_cloud as pywren
from random import random
from time import time
from math import exp
from scipy.stats import norm
from numpy import median

ACTIONS = 5000
TOTAL = ACTIONS
print("Total Random Samples is: " + str(TOTAL))

gold = {"drift": 0.000142559, "std_dev": 0.010561899, "last_price": 1296.5, "days_2_predict": 219}
mlnx = {"drift": 0.000581742829, "std_dev": 0.029879238, "last_price": 64.7, "days_2_predict": 219}
ibm = {"drift": 0.000091967236, "std_dev": 0.012404562, "last_price": 153.42, "days_2_predict": 219}
nvda = {"drift": 0.000936809, "std_dev": 0.027145343, "last_price": 193.5, "days_2_predict": 219}

STOCK_DATA = nvda

iterdata = [[]] * ACTIONS


def my_map_function():
    std_dev = STOCK_DATA["std_dev"]
    drift = STOCK_DATA["drift"]
    last_price = STOCK_DATA["last_price"]
    days_2_predict = STOCK_DATA["days_2_predict"]
    predicts_est = [0] * (days_2_predict + 1)
    predicts_est[0] = last_price
    for predict in range(1, days_2_predict + 1):
        rand = random()
        pow_r = norm.ppf(rand)
        predicts_est[predict] = predicts_est[predict - 1] * exp(drift + (std_dev * pow_r))
    return predicts_est

def my_reduce_function(list_of_lists):
    return [median(x) for x in zip(*list_of_lists)]


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""

FLAG = "LOCAL"
# FLAG = "CLOUD"
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
