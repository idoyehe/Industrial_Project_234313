from random import random
from PwrenUtils.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN

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


def my_reduce_function(results, futures):
    sumPI = 0
    for map_result in results:
        for p in map_result:
            sumPI += p
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": float(4 * (sumPI / TOTAL))}


executor = ExecutorWrap(ACTIONS, "Pi_monte_carlo_" + str(PER_ACTION * ACTIONS))
executor.set_location(exe_location)
result_obj = executor.map_reduce_execution(my_map_function, iterdata, my_reduce_function)
print("Amortized PI is: ")
print(result_obj)
