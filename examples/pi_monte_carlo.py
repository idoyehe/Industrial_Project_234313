"""
Simple PyWren example using the map_reduce method.

In this example the map_reduce() method will launch one
map function for each entry in 'iterdata', and then it will
wait locally for the results. Once the results be ready, it
will launch the reduce function.
"""
import pywren_ibm_cloud as pywren
from random import random

ACTIONS = 55
PER_ACTION = 1009000

iterdata = [False] * ACTIONS


def my_map_function(curr):
    points = []
    for i in range(PER_ACTION):
        x = random()
        y = random()
        points.append((x ** 2) + (y ** 2) <= 1)
    return points


def my_reduce_function(results):
    in_circle = 0
    num_samples = 0
    for map_result in results:
        for res in map_result:
            in_circle = in_circle + int(res)
            num_samples += 1
    return float(4 * (in_circle / num_samples))


"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""
# for i in range(len(iterdata)):
#     iterdata[i] = my_map_function(iterdata[i])
# print(my_reduce_function(iterdata))

pw = pywren.ibm_cf_executor()
pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
print(pw.get_result())
