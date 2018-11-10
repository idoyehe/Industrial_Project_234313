"""
Simple PyWren example using the map_reduce method.

In this example the map_reduce() method will launch one
map function for each entry in 'iterdata', and then it will
wait locally for the results. Once the results be ready, it
will launch the reduce function.
"""
import pywren_ibm_cloud as pywren
from random import random
iterdata = [False] * 10


def my_map_function(curr):
    x = random() - 0.5
    y = random() - 0.5
    return (x ** 2) + (y ** 2) <= 0.25


def my_reduce_function(results):
    in_circle = 0
    for map_result in results:
        in_circle = in_circle + int(map_result)
    return float(4 * (in_circle / len(results)))

"""
Set 'reducer_wait_local=False' to launch the reducer and wait for
the results remotely.
"""
pw = pywren.ibm_cf_executor()
pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
print(pw.get_result())
