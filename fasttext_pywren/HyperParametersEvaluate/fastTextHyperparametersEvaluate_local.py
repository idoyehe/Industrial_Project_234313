import fastText as fstTxt
from fasttext_pywren.HyperParametersEvaluate.localK_fold_cross_validation import *

models_names = {"dbpedia": "../Models/dbpedia/dbpedia.train",
                "yelp": "../Models/yelp_review_full/yelp_review_full.train"}


def fastText_evaluate(train_path, test_path, hyperparameters_set):
    to_valid_model = fstTxt.train_supervised(train_path, **hyperparameters_set)
    result = to_valid_model.test(test_path)
    return {"precision": result[1], "recall": result[2]}


lkf = LocalKFoldCrossValidation(5, models_names['dbpedia'], ("precision", "recall"), fastText_evaluate)
results = lkf.k_fold_cross_validation()
print(results)
