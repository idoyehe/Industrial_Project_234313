from random import random
from PwrenUtils.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN

MAP_INSTANCES = 10


class EstimatePI:
    randomize_per_map = 10000000

    def __init__(self):
        self.total_randomize_points = MAP_INSTANCES * self.randomize_per_map

    def __str__(self):
        return "Total Randomize Points: {:,}".format(self.randomize_per_map)

    @staticmethod
    def predicate():
        x = random()
        y = random()
        return (x ** 2) + (y ** 2) <= 1

    def randomize_points(self, data):
        in_circle = 0
        for _ in range(self.randomize_per_map):
            in_circle += self.predicate()
        return float(in_circle / self.randomize_per_map)

    def process_in_circle_points(self, results, futures):
        in_circle_percent = 0
        for map_result in results:
            in_circle_percent += map_result
        run_statuses = [f.run_status for f in futures]
        invoke_statuses = [f.invoke_status for f in futures]
        estimate_PI = float(4 * (in_circle_percent / MAP_INSTANCES))
        return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": estimate_PI}


est_pi = EstimatePI()
iterdata = [0] * MAP_INSTANCES
executor = ExecutorWrap(MAP_INSTANCES, "Pi_monte_carlo_" + str(est_pi))
executor.set_location(exe_location)
result_obj = executor.map_reduce_execution(est_pi.randomize_points, iterdata, est_pi.process_in_circle_points)
print(str(est_pi))
print("Amortized PI is: ")
print(result_obj)
