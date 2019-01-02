import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywrenGenericHyperparameterEvaluate

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}

bucketname = 'fasttext-train-bucket'

"""example"""
iter_parameters = \
    [{'lr': 0.3, 'lrUpdateRate': 80, 'ws': 6, 'epoch': 5}]

K = 1

"""evaluation model function instance"""
def fastText_evaluate(parameters_dict, train_path, test_path):
    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


"""call for PyWren exaction with list of hyperparameters"""
hyperparameters = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(fastText_evaluate, bucketname,
                                                                               files_names["dbpedia"],
                                                                               job_name="fastText_hyperparameters_evaluate_yelp",
                                                                               graphs_path="../../InvocationsGraphsFiles/")
hyperparameters.set_kvalue(K)
hyperparameters.set_parameters(iter_parameters)
print(hyperparameters.evaluate_params(runtime="fasttext-hyperparameter"))
