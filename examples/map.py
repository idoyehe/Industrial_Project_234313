"""
Simple PyWren example using the map method.

In this example the map() method will launch one
map function for each entry in 'iterdata'. Finally
it will print the results for each invocation with
pw.get_all_result()
"""
import pywren_ibm_cloud as pywren
from os import mkdir, path
from shutil import rmtree


class Ido(object):
    def __init__(self):
        self.iterdata = [1]

    def my_map_function(self, x):
        def check_files():
            return path.exists("/47") and path.exists("/48")

        rmtree('/47/', ignore_errors=True)
        rmtree('/48/', ignore_errors=True)
        mkdir('/47/')
        mkdir('/48/')
        sub_pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameters")
        sub_pw.call_async(check_files, [])
        return sub_pw.get_result()


ido = Ido()

pw = pywren.ibm_cf_executor(runtime="fasttext-hyperparameters")
pw.map(ido.my_map_function, [1, 1])
print(pw.get_result())
