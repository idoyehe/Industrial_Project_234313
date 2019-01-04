import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywrenGenericHyperparameterEvaluate

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}

bucketname = 'fasttext-train-datasets'

"""example"""
iter_parameters = \
    [[{'lr': 1, 'lrUpdateRate': 20, 'ws': 6, 'epoch': 12}], [{'lr': 0.6, 'lrUpdateRate': 70, 'ws': 7, 'epoch': 9}],
     [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 7, 'epoch': 13}], [{'lr': 0.1, 'lrUpdateRate': 100, 'ws': 7, 'epoch': 13}],
     [{'lr': 0.9, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}], [{'lr': 0.7, 'lrUpdateRate': 20, 'ws': 4, 'epoch': 13}],
     [{'lr': 0.9, 'lrUpdateRate': 20, 'ws': 6, 'epoch': 8}], [{'lr': 0.3, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 7}],
     [{'lr': 0.2, 'lrUpdateRate': 40, 'ws': 3, 'epoch': 13}], [{'lr': 0.3, 'lrUpdateRate': 40, 'ws': 5, 'epoch': 11}],
     [{'lr': 0.1, 'lrUpdateRate': 20, 'ws': 3, 'epoch': 15}], [{'lr': 0.8, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 6}],
     [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 10}], [{'lr': 1, 'lrUpdateRate': 100, 'ws': 5, 'epoch': 12}],
     [{'lr': 0.5, 'lrUpdateRate': 90, 'ws': 3, 'epoch': 12}], [{'lr': 0.6, 'lrUpdateRate': 80, 'ws': 3, 'epoch': 9}],
     [{'lr': 0.6, 'lrUpdateRate': 100, 'ws': 3, 'epoch': 15}], [{'lr': 0.2, 'lrUpdateRate': 100, 'ws': 7, 'epoch': 11}],
     [{'lr': 0.3, 'lrUpdateRate': 40, 'ws': 6, 'epoch': 5}], [{'lr': 0.9, 'lrUpdateRate': 20, 'ws': 3, 'epoch': 10}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 13}], [{'lr': 0.2, 'lrUpdateRate': 30, 'ws': 7, 'epoch': 12}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}], [{'lr': 1, 'lrUpdateRate': 90, 'ws': 5, 'epoch': 12}],
     [{'lr': 0.7, 'lrUpdateRate': 50, 'ws': 3, 'epoch': 8}], [{'lr': 0.6, 'lrUpdateRate': 30, 'ws': 6, 'epoch': 10}],
     [{'lr': 0.4, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 15}], [{'lr': 0.5, 'lrUpdateRate': 30, 'ws': 3, 'epoch': 11}],
     [{'lr': 0.5, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 15}], [{'lr': 0.5, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 7}],
     [{'lr': 0.5, 'lrUpdateRate': 60, 'ws': 3, 'epoch': 14}], [{'lr': 0.7, 'lrUpdateRate': 70, 'ws': 5, 'epoch': 15}],
     [{'lr': 0.5, 'lrUpdateRate': 90, 'ws': 3, 'epoch': 13}], [{'lr': 0.4, 'lrUpdateRate': 80, 'ws': 6, 'epoch': 6}],
     [{'lr': 0.7, 'lrUpdateRate': 70, 'ws': 4, 'epoch': 13}], [{'lr': 0.6, 'lrUpdateRate': 40, 'ws': 6, 'epoch': 12}],
     [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 14}], [{'lr': 0.2, 'lrUpdateRate': 50, 'ws': 6, 'epoch': 5}],
     [{'lr': 1, 'lrUpdateRate': 90, 'ws': 7, 'epoch': 13}], [{'lr': 1, 'lrUpdateRate': 70, 'ws': 3, 'epoch': 7}],
     [{'lr': 0.2, 'lrUpdateRate': 20, 'ws': 7, 'epoch': 9}], [{'lr': 0.3, 'lrUpdateRate': 100, 'ws': 5, 'epoch': 7}],
     [{'lr': 0.8, 'lrUpdateRate': 90, 'ws': 7, 'epoch': 10}], [{'lr': 0.2, 'lrUpdateRate': 20, 'ws': 4, 'epoch': 11}],
     [{'lr': 1, 'lrUpdateRate': 50, 'ws': 7, 'epoch': 5}], [{'lr': 0.5, 'lrUpdateRate': 40, 'ws': 4, 'epoch': 9}],
     [{'lr': 0.9, 'lrUpdateRate': 80, 'ws': 7, 'epoch': 9}], [{'lr': 0.4, 'lrUpdateRate': 40, 'ws': 4, 'epoch': 14}],
     [{'lr': 0.2, 'lrUpdateRate': 60, 'ws': 4, 'epoch': 15}], [{'lr': 0.9, 'lrUpdateRate': 40, 'ws': 7, 'epoch': 14}]]

K = 3

"""evaluation model function instance"""
def fastText_evaluate(parameters_dict, train_path, test_path):
    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


"""call for PyWren exaction with list of hyperparameters"""
hyperparameters = pywrenGenericHyperparameterEvaluate.PywrenHyperParameterUtil(fastText_evaluate, bucketname,
                                                                               files_names["dbpedia"],
                                                                               job_name="fastText_hyperparameters_evaluate_dbpedia",
                                                                               graphs_path="../../InvocationsGraphsFiles/")
hyperparameters.set_kvalue(K)
hyperparameters.set_parameters(iter_parameters)
print(hyperparameters.evaluate_params(runtime="fasttext-hyperparameter"))
