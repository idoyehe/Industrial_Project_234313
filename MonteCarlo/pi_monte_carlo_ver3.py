from random import random
from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN

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


def my_reduce_function(results, futures):
    sumPI = 0
    for map_result in results:
        sumPI += map_result
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": float(4 * (sumPI / ACTIONS))}


executor = ExecutorWrap(ACTIONS, "Pi_monte_carlo_" + str(PER_ACTION * ACTIONS))
result_obj = executor.map_reduce_execution(my_map_function, iterdata, my_reduce_function)
print("Amortized PI is: ")
print(result_obj)
