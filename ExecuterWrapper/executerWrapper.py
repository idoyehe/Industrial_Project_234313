from enum import Enum
from time import time
import pywren_ibm_cloud as pywren
from pickle import dump


class Location(Enum):
    LOCAL = 1
    CLOUD = 2


class ExecuterWrap(object):
    def __init__(self):
        self.execution_location = Location.LOCAL

    def set_location(self, new_lcl=Location.LOCAL):
        self.execution_location = new_lcl

    def map_reduce_execution(self, my_map_function, iterdata, my_reduce_function):
        if self.execution_location == Location.LOCAL:
            action_number = len(iterdata)
            start_time = time()
            for i in range(action_number):
                iterdata[i] = my_map_function(iterdata[i])
            result_object = my_reduce_function(iterdata)
        else:
            start_time = time()
            pw = pywren.ibm_cf_executor()
            future = pw.map_reduce(my_map_function, iterdata, my_reduce_function, reducer_wait_local=False)
            result_object = pw.get_result()
            run_statuses = future.run_status
            invoke_statuses = future.invoke_status
            res = {'run_statuses': run_statuses, 'invoke_statuses': invoke_statuses}
            dump(res, open('statuses.pickle', 'wb'), -1)

        elapsed = time()
        print("\nDuration: " + str(elapsed - start_time) + " Sec")
        return result_object
