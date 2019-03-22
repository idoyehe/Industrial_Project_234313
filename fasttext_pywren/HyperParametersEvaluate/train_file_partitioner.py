from os import path
from sklearn.model_selection import KFold


class KFoldCrossPartitioner(object):
    def __init__(self, k_value: int, train_file_path: str, output_path: str, model_name: str):
        self._k_value = k_value
        self._train_file_path = train_file_path
        self._output_path = output_path
        self._model_name = model_name

    def k_fold_cross_partitioner(self):
        if not (path.exists(self._train_file_path) and path.exists(self._output_path)):
            raise FileNotFoundError("Provided paths sre not exists in the local filesystem\n")

        train_file = open(self._train_file_path, 'r')
        labeled_data = train_file.read().splitlines()
        kf = KFold(n_splits=self._k_value, shuffle=True)
        splits = kf.split(labeled_data)

        fold_prefix: str = self._model_name + "_fold_"
        fold_index: int = 0

        for train_index, test_index in splits:
            train_lines = [labeled_data[index] + "\n" for index in train_index]
            test_lines = [labeled_data[index] + "\n" for index in test_index]

            train_path: str = self._output_path + "/" + fold_prefix + str(fold_index) + ".train"
            test_path: str = self._output_path + "/" + fold_prefix + str(fold_index) + ".test"
            train_file = open(train_path, 'w')
            test_file = open(test_path, 'w')

            train_file.writelines(train_lines)
            test_file.writelines(test_lines)
            test_file.close()
            train_file.close()
            fold_index += 1

        train_file.close()
