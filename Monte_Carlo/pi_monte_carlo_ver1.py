from random import random
from ExecuterWrapper.executerWrapper import ExecuterWrap, Location

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


executer = ExecuterWrap()
executer.set_location(Location.CLOUD)
result_obj = executer.map_reduce_execution(my_map_function, iterdata, my_reduce_function)
print("Amortized PI is: ")
print(result_obj)
