import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate import pywren_hyperparameter

files_names = {"dbpedia": "dbpedia.train",
               "yelp": "yelp_review_full.train"}

bucketname = 'fasttext-train-bucket'


iter_parameters = [{
    "lr": 0.6,
    "dim": 100,
    "ws": 5,
    "epoch": 7,
    "minCount": 1},
    {"lr": 0.1,
     "dim": 100,
     "ws": 5,
     "epoch": 1,
     "minCount": 1},
    {"lr": 1,
     "dim": 100,
     "ws": 5,
     "epoch": 2,
     "minCount": 1}
]


def fastText_evaluate(parameters_dict, train_path, test_path):
    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


hyperparameters = pywren_hyperparameter.PywrenHyperParameterUtil(fastText_evaluate, bucketname, files_names["dbpedia"])
hyperparameters.set_kvalue(5)
hyperparameters.set_parameters(iter_parameters)
print(hyperparameters.evaluate_params(runtime="fasttext-hyperparameter"))




