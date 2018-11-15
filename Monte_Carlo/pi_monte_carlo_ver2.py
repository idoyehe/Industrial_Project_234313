import pywren_ibm_cloud as pywren
from random import random
from time import time
from pickle import dump

ACTIONS = 50
PER_ACTION = 1000000
TOTAL = ACTIONS * PER_ACTION
print("Total points is: " + str(TOTAL))

iterdata = [0] * ACTIONS


def my_map_function(curr):
    list_points = []
    for i in range(PER_ACTION):
        x = random()
        y = random()
        list_points.append((x ** 2) + (y ** 2) <= 1)
    return list_points


def my_reduce_function(results):
    sumPI = 0
    for map_result in results:
        for p in map_result:
            sumPI += p
    return float(4 * (sumPI / TOTAL))

"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""
FLAG = "LOCAL"
if FLAG == "LOCAL":
    start_time = time()
    for i in range(len(iterdata)):
        iterdata[i] = my_map_function(iterdata[i])
    print("Amortized PI is: ")
    print(my_reduce_function(iterdata))
else:
    start_time = time()
    pw = pywren.ibm_cf_executor()
    future = pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
    PI = pw.get_result()
    run_statuses = future.run_status
    invoke_statuses = future.invoke_status
    res = {'run_statuses': run_statuses, 'invoke_statuses': invoke_statuses}
    dump(res, open('statuses.pickle', 'wb'), -1)
    print("Amortized PI is: ")
    print(PI)
elapsed = time()
print("\nDuration: " + str(elapsed - start_time) + " Sec")
