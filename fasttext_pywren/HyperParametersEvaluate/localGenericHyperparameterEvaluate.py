from sklearn.model_selection import KFold
from os import mkdir
from shutil import rmtree
from multiprocessing import Pool
from time import time


def _createValidationFiles(_labeled_data, _train_index, _test_indexes, _train_file_name, _test_file_name, _path: str):
    mkdir(_path)
    train_path = _path + _train_file_name
    test_path = _path + _test_file_name
    train_file = open(train_path, 'w')
    test_file = open(test_path, 'w')
    for index in _train_index:
        train_file.writelines(_labeled_data[index] + "\n")
    for index in _test_indexes:
        test_file.writelines(_labeled_data[index] + "\n")
    test_file.close()
    train_file.close()


class LocalKFoldCrossValidation(object):
    def __init__(self, k_value: int, train_file: str, evaluate_keys, evaluate_function):
        self.k_value: int = k_value
        self.train_file: str = train_file
        self.evaluate_function = evaluate_function
        self.evaluate_keys = evaluate_keys

        self.folders_prefix = "./validation_"
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

        p = Pool(self.k_value)
        p.starmap(func=_createValidationFiles, iterable=args_list)
        opened_train_file.close()

    def __cleaner(self):
        for i in range(self.k_value):
            rmtree(self.folders_prefix + str(i), ignore_errors=True)

    def __reducer_average_validator(self, results):
        k_cross_valid_dict = {}
        for key in self.eval_keys:  # initialize all sums to 0
            k_cross_valid_dict[key] = 0

        for res in results:
            for key in self.eval_keys:  # adding to all sums
                k_cross_valid_dict[key] += res[key]

        for key in self.eval_keys:  # average each sum
            k_cross_valid_dict[key] /= len(results)
        return k_cross_valid_dict

    def k_fold_cross_validation(self, hyperparameters_set={}):
        self.__cleaner()  # make sure last session cleaned
        start_time = time()
        self.__k_partitioner()
        args_list = []
        for eval_index in range(self.k_value):
            path: str = self.folders_prefix + str(eval_index) + "/"
            train_path: str = path + self.train_file_name
            test_path: str = path + self.test_file_name
            args_list.append((train_path, test_path, hyperparameters_set))

        p = Pool(self.k_value)
        results = p.starmap(func=self.evaluate_function, iterable=args_list)
        results = self.__reducer_average_validator(results)
        results['time'] = time() - start_time
        self.__cleaner()  # make sure last session cleaned
        return results
