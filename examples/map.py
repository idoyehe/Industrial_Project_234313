"""
Simple PyWren example using the map method.

In this example the map() method will launch one
map function for each entry in 'iterdata'. Finally
it will print the results for each invocation with
pw.get_all_result()
"""
import pywren_ibm_cloud as pywren

class Ido(object):
    def __init__(self):
        self.iterdata = [(1, {"ido": 1}), (2, {"ido": 1}), (3, {"ido": 1}), (4, {"ido": 1})]
        self.A = 7

    def my_map_function(self, x, y):
        return x + self.A, y

ido = Ido()

pw = pywren.ibm_cf_executor()
pw.map(ido.my_map_function, ido.iterdata)
print(pw.get_result())
