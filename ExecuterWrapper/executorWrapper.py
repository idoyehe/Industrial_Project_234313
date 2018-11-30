from enum import Enum
from time import time
import pywren_ibm_cloud as pywren
from pickle import dump
import logging


class Location(Enum):
    LOCAL = 1
    PYWREN = 2


class ExecutorWrap(object):
    def __init__(self, total_actions):
        self.total_actions = total_actions
        self.execution_location = Location.LOCAL

    def set_location(self, new_lcl=Location.LOCAL):
        self.execution_location = new_lcl

    def _local_execution(self, map_function, iterable_data, reduce_function):
        start_time = time()
        for i in range(self.total_actions):
            iterable_data[i] = map_function(iterable_data[i])
        result_object = reduce_function(iterable_data)
        elapsed = time()
        return result_object['results'], elapsed - start_time

    def _pywren_execution(self, map_function, iterable_data, reduce_function):
        logging.basicConfig(level=logging.DEBUG)
        start_time = time()
        pw = pywren.ibm_cf_executor()
        pw.map_reduce(map_function, iterable_data, reduce_function, reducer_wait_local=False)
        result_object = pw.get_result()
        elapsed = time()
        futures = result_object['futures']
        if futures is not None:
            run_statuses = [f.run_status for f in futures]
            invoke_statuses = [f.invoke_status for f in futures]
            res = {'run_statuses': run_statuses, 'invoke_statuses': invoke_statuses}
            dump(res, open('./statuses.pickle', 'wb'), -1)
        pw.clean()
        return result_object['results'], elapsed - start_time


    def map_reduce_execution(self, map_function, iterable_data, reduce_function):

        if self.execution_location == Location.LOCAL:
            result_object, duration = self._local_execution(map_function, iterable_data, reduce_function)
        else:
            assert self.execution_location == Location.PYWREN
            result_object, duration = self._pywren_execution(map_function, iterable_data, reduce_function)

        print("\nDuration: " + str(duration) + " Sec")
        return result_object
