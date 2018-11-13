import pywren_ibm_cloud as pywren
from random import random
from time import time
ACTIONS = 10000
PER_ACTION = 1
print("Total points is: " + str(PER_ACTION * ACTIONS))

iterdata = [0] * ACTIONS


def my_map_function(curr):
    x = random()
    y = random()
    return (x ** 2) + (y ** 2) <= 1

def my_reduce_function(results):
    total_in = 0
    for map_result in results:
        total_in += map_result
    return float(4 * (total_in / ACTIONS))


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""

FLAG = ""
if FLAG == "LOCAL":
    start_time = time()
    for i in range(len(iterdata)):
        iterdata[i] = my_map_function(iterdata[i])
    print("Amortized PI is: ")
    print(my_reduce_function(iterdata))
else:
    start_time = time()
    pw = pywren.ibm_cf_executor()
    pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
    PI = pw.get_result()
    print("Amortized PI is: ")
    print(PI)
elapsed = time()
print("\nDuration: " + str(elapsed - start_time) + " Sec")
