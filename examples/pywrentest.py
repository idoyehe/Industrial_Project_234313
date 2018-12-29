"""
Simple PyWren example using one single function invocation
"""
import pywren_ibm_cloud as pywren


def my_function(x):
    return x * 2


def my_reduce_function(results):
    return sum(results)


ido = [1, 2]


def my_function_wrap(data):
    data = [data] * 2
    pw = pywren.ibm_cf_executor()
    pw.map_reduce(my_function, data, my_reduce_function)
    return pw.get_result()


if __name__ == '__main__':
    pw = pywren.ibm_cf_executor()
    pw.map(my_function_wrap, ido)
    print(pw.get_result())
    pw.create_timeline_plots(dst="../InvocationsGraphsFiles/", name='testmap')
