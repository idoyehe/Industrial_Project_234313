from enum import Enum
from time import time
import pywren_ibm_cloud as pywren
import logging


class Location(Enum):
    LOCAL = 1
    PYWREN = 2
    PYWREN_DEBUG = 3


class ExecutorWrap(object):
    def __init__(self, total_actions, invocation_name, graphs_path=None):
        self.total_actions = total_actions
        self.execution_location = Location.LOCAL
        self.invocation_name = invocation_name
        self.graphs_path = graphs_path if graphs_path else "../InvocationsGraphsFiles/"
        self.duration = None

    def set_location(self, new_lcl=Location.LOCAL):
        self.execution_location = new_lcl

    def _local_execution(self, map_function, iterable_data, reduce_function):
        start_time = time()
        for i in range(self.total_actions):
            iterable_data[i] = map_function(*iterable_data[i])
        result_object = reduce_function(iterable_data, [])
        elapsed = time()
        return result_object['results'], elapsed - start_time

    def _pywren_execution(self, map_function, iterable_data, reduce_function, chunk_size, runtime):
        if self.execution_location == Location.PYWREN_DEBUG:
            logging.basicConfig(level=logging.DEBUG)
        start_time = time()
        pw = pywren.ibm_cf_executor(runtime=runtime)
        pw.map_reduce(map_function, iterable_data, reduce_function, chunk_size=chunk_size, reducer_wait_local=False)
        result_object = pw.get_result()
        elapsed = time()
        if result_object.get('run_statuses', False) and result_object.get('invoke_statuses', False):
            pw.create_timeline_plots(dst=self.graphs_path, name=self.invocation_name,
                                     run_statuses=result_object['run_statuses'], invoke_statuses=result_object['invoke_statuses'])
        pw.clean()
        return result_object['results'], elapsed - start_time

    def map_reduce_execution(self, map_function, iterable_data, reduce_function, chunk_size=None, runtime="pywren_3.6"):

        if self.execution_location == Location.LOCAL:
            result_object, duration = self._local_execution(map_function, iterable_data, reduce_function)
        else:
            assert self.execution_location == Location.PYWREN or self.execution_location == Location.PYWREN_DEBUG
            result_object, duration = self._pywren_execution(map_function, iterable_data, reduce_function, chunk_size, runtime)

        self.duration = duration
        print("\nDuration: " + str(duration) + " Sec")
        return result_object

    def get_last_duration(self):
        return self.duration
