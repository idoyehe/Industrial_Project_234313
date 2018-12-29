import fastText as fstTxt

from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN
K = 2
GIVEN_PARAMS = {
    "lr": 0.1,
    "dim": 100,
    "ws": 5,
    "epoch": 5,
    "minCount": 1}

k_list = [(i, GIVEN_PARAMS) for i in range(K)]


def map_k_cross_validation(valid_index, parameters_dict):
    source_path = "/fasttext/validation/dbpedia.train"
    valid_path = "/fasttext/validation/source.vaild"
    train_path = "/fasttext/validation/source.train"

    input_file = open(source_path, 'r').read()
    train_file = open(train_path, 'w')
    vaild_file = open(valid_path, 'w')

    current_index = 0
    for line in input_file.splitlines():
        line += "\n"
        if current_index % K == valid_index:
            vaild_file.write(line)
        else:
            train_file.write(line)

        current_index += 1

    vaild_file.close()
    train_file.close()

    to_valid_model = fstTxt.train_supervised(train_path, **parameters_dict)
    result = to_valid_model.test(valid_path)
    return {"precision": result[1], "recall": result[2]}


def reducer_average_validator(results, futures):
    avg_precision = 0
    avg_recall = 0

    for current in results:
        avg_precision += current["precision"]
        avg_recall += current["recall"]

    avg_precision /= K
    avg_recall /= K
    avg_result = {"precision": avg_precision, "recall": avg_recall}
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": avg_result}


executor = ExecutorWrap(K, "K cross validation" + str(K))
executor.set_location(exe_location)
result_obj = executor.map_reduce_execution(map_k_cross_validation, k_list, reducer_average_validator,
                                           runtime="fasttext-hyperparameter")

print("K cross validation result: ", result_obj)
