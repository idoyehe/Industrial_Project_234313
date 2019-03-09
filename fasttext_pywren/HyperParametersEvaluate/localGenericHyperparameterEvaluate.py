from sklearn.model_selection import KFold
from os import mkdir
from shutil import rmtree
import billiard as multiprocessing
from time import time


def _createValidationFiles(_labeled_data: list, _train_index: list, _test_index: list, _train_file_name: str, _test_file_name: str,
                           _path: str):
    mkdir(_path)
    train_path = _path + _train_file_name
    test_path = _path + _test_file_name
    train_lines = [_labeled_data[index] + "\n" for index in _train_index]
    test_lines = [_labeled_data[index] + "\n" for index in _test_index]
    train_file = open(train_path, 'w')
    test_file = open(test_path, 'w')
    train_file.writelines(train_lines)
    test_file.writelines(test_lines)
    test_file.close()
    train_file.close()


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


def _evaluate_single_hyperparameters(k_value,
                                     folders_prefix,
                                     train_file_name,
                                     test_file_name,
                                     evaluate_function,
                                     evaluate_keys,
                                     hyperparameters_set={}):
    args_list = []
    for eval_index in range(k_value):
        path: str = folders_prefix + str(eval_index) + "/"
        train_path: str = path + train_file_name
        test_path: str = path + test_file_name
        args_list.append((train_path, test_path, hyperparameters_set))

    p = multiprocessing.Pool(k_value)
    results = p.starmap(func=evaluate_function, iterable=args_list)
    results = _reducer_average_validator(evaluate_keys, results)
    return results


class LocalKFoldCrossValidation(object):
    def __init__(self, k_value: int, train_file: str, evaluate_keys, evaluate_function):
        self.k_value: int = k_value
        self.train_file: str = train_file
        self.evaluate_function = evaluate_function
        self.evaluate_keys = evaluate_keys

        self.folders_prefix = "./fold_"
        self.train_file_name = "database.train"
        self.test_file_name = "database.test"

    def __k_partitioner(self):
        opened_train_file = open(self.train_file, 'r')
        labeled_data = opened_train_file.read().splitlines()
        kf = KFold(n_splits=self.k_value, shuffle=True)
        splits = kf.split(labeled_data)
        i = 0

        args_list = []
        for train_index, test_index in splits:
            path: str = self.folders_prefix + str(i) + "/"
            i += 1
            args_list.append((labeled_data,
                              train_index,
                              test_index,
                              self.train_file_name,
                              self.test_file_name,
                              path))

        p = multiprocessing.Pool(self.k_value)
        p.starmap(func=_createValidationFiles, iterable=args_list)
        opened_train_file.close()

    def __cleaner(self):
        for i in range(self.k_value):
            rmtree(self.folders_prefix + str(i), ignore_errors=True)

    def hyperparameters_kfc(self, hyperparameters_sets=[[{}],[{}]]):
        self.__cleaner()  # make sure last session cleaned
        start_time = time()
        self.__k_partitioner()
        args_list = []
        for hyp_set in hyperparameters_sets:
            args_list.append((self.k_value,
                              self.folders_prefix,
                              self.train_file_name,
                              self.test_file_name,
                              self.evaluate_function,
                              self.evaluate_keys,
                              hyp_set[0]))

        p = multiprocessing.Pool(len(hyperparameters_sets))
        results = p.starmap(func=_evaluate_single_hyperparameters, iterable=args_list)
        completion_time = time() - start_time
        self.__cleaner()  # make sure current session cleaned
        return {"results": results, "completion_time": completion_time}
