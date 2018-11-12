import pywren_ibm_cloud as pywren
from random import random
from time import time
ACTIONS = 1000
PER_ACTION = 10000000
print("Total points is: " + str(PER_ACTION * ACTIONS))

iterdata = [0] * ACTIONS


def my_map_function(curr):
    total_in_circle = 0
    for i in range(PER_ACTION):
        x = random()
        y = random()
        total_in_circle += ((x ** 2) + (y ** 2) <= 1)
    return float(total_in_circle / PER_ACTION)


def my_reduce_function(results):
    sumPI = 0
    for map_result in results:
        sumPI += map_result
    return float(4 * (sumPI / ACTIONS))


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""
start_time = time()
# for i in range(len(iterdata)):
#     iterdata[i] = my_map_function(iterdata[i])
# print("Amortized PI is: ")
# print(my_reduce_function(iterdata))

pw = pywren.ibm_cf_executor()
pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
PI = pw.get_result()
print("Amortized PI is: ")
print(PI)
elapsed = time()
print("\nDuration: " + str(elapsed - start_time) + " Sec")
