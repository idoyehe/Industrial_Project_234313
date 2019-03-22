import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywrenGenericHyperparameterEvaluate
from fasttext_pywren.HyperParametersEvaluate.randomHyperparameters import random_search

bucket_name = 'fasttext-train-datasets'

"""evaluation model function instance"""
def fastText_evaluate(train_path, test_path, hyperparameters_set):
    from time import time
    start = time()
    to_valid_model = fstTxt.train_supervised(train_path, **hyperparameters_set)
    result = to_valid_model.test(test_path)
    end = time()
    return {"precision": result[1], "recall": result[2], "cpu_time": end - start}


"""call for PyWren exaction with list of hyperparameters"""
hyperparameters = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(fastText_evaluate,
                                                                               bucket_name,
                                                                               "ag_news",
                                                                               job_name="fastText_hyperparameters_evaluate_ag_news",
                                                                               local_graphs_path="../../InvocationsGraphsFiles/")
hyperparameters.set_kvalue(5)
hyperparameters.set_evaluation_keys(("precision", "recall", "cpu_time"))
hyperparameters.set_parameters(random_search(10))
print(hyperparameters.evaluate_params(runtime="fasttext-hyperparameters"))
