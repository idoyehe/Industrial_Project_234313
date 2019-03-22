import billiard as multiprocessing
from time import time


def _reducer_average_validator(evaluate_keys, results):
    k_cross_valid_dict = {}
    for key in evaluate_keys:  # initialize all sums to 0
        k_cross_valid_dict[key] = 0

    for res in results:
        for key in evaluate_keys:  # adding to all sums
            k_cross_valid_dict[key] += res[key]

    for key in evaluate_keys:  # average each sum
        k_cross_valid_dict[key] /= len(results)
    return k_cross_valid_dict


def _evaluate_single_hyperparameters_parallel(k_value,
                                              folder_path,
                                              model_name,
                                              evaluate_function,
                                              evaluate_keys,
                                              hyperparameters_set={}):
    args_list = []
    for eval_index in range(k_value):
        path: str = folder_path
        train_path: str = path + model_name + "_fold_" + str(eval_index) + ".train"
        test_path: str = path + model_name + "_fold_" + str(eval_index) + ".test"
        args_list.append((train_path, test_path, hyperparameters_set))

    p = multiprocessing.Pool(k_value)
    results = p.starmap(func=evaluate_function, iterable=args_list)
    results = _reducer_average_validator(evaluate_keys, results)
    return results


class LocalKFoldCrossValidation(object):
    def __init__(self, k_value: int, model_name: str, evaluate_keys, evaluate_function):
        self.k_value: int = k_value
        self.model_name: str = model_name
        self.evaluate_function = evaluate_function
        self.evaluate_keys = evaluate_keys
        self.folder_path = "./folds_" + model_name + "/"

    def _evaluate_single_hyperparameters_serial(self, hyperparameters_set={}):
        start_time = time()
        results_list = []
        for eval_index in range(self.k_value):
            path: str = self.folder_path
            train_path: str = path + self.model_name + "_fold_" + str(eval_index) + ".train"
            test_path: str = path + self.model_name + "_fold_" + str(eval_index) + ".test"
            results_list.append(self.evaluate_function(train_path, test_path, hyperparameters_set))

        results = _reducer_average_validator(self.evaluate_keys, results_list)
        results['validation_completion_time'] = time() - start_time
        return results

    def hyperparameters_kfc_serial(self, hyperparameters_sets=[[{}]]):
        start_time = time()
        results = []
        for hyp_set in hyperparameters_sets:
            results.append(self._evaluate_single_hyperparameters_serial(hyp_set[0]))
        total_completion_time = time() - start_time
        return {"results": results, "total_completion_time": total_completion_time}

    def hyperparameters_kfc_parallel(self, hyperparameters_sets=[[{}]]):
        start_time = time()
        args_list = []
        for hyp_set in hyperparameters_sets:
            args_list.append((self.k_value,
                              self.folder_path,
                              self.model_name,
                              self.evaluate_function,
                              self.evaluate_keys,
                              hyp_set[0]))

        p = multiprocessing.Pool(len(hyperparameters_sets))
        results = p.starmap(func=_evaluate_single_hyperparameters_parallel, iterable=args_list)
        total_completion_time = time() - start_time
        return {"results": results, "total_completion_time": total_completion_time}
