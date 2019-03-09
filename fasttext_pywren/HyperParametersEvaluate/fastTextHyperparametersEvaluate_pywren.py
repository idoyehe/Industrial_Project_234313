import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywrenGenericHyperparameterEvaluate

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}

bucketname = 'fasttext-train-datasets'

"""example"""
iter_parameters = [0] * 2

K = 5

"""evaluation model function instance"""


def fastText_evaluate(train_path, test_path, parameters_dict={}):
    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


"""call for PyWren exaction with list of hyperparameters"""
hyperparameters = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(fastText_evaluate, bucketname,
                                                                               files_names["dbpedia"],
                                                                               job_name="fastText_hyperparameters_evaluate_dbpedia",
                                                                               graphs_path="../../InvocationsGraphsFiles/")
hyperparameters.set_kvalue(K)
hyperparameters.set_evaluation_keys(["precision", "recall"])
hyperparameters.set_parameters(iter_parameters)
print(hyperparameters.evaluate_params(runtime="fasttext-hyperparameters"))
