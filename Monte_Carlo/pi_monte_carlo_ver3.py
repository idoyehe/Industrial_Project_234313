from random import random
from ExecuterWrapper.executerWrapper import ExecuterWrap, Location

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


executer = ExecuterWrap()
executer.set_location(Location.CLOUD)
result_obj = executer.map_reduce_execution(my_map_function, iterdata, my_reduce_function)
print("Amortized PI is: ")
print(result_obj)
