import fastText as fstTxt
from ExecuterWrapper.executorWrapper import ExecutorWrap, Location

exe_location = Location.PYWREN

ag_news_model = "/fasttext/models/ag_news.ftz"  # from docker
dbpedia_model = "/fasttext/models/dbpedia.ftz"  # from docker
yelp_review_model = "/fasttext/models/yelp_review_full.ftz"  # from docker
sogou_news_model = "/fasttext/models/sogou_news.ftz"  # from docker

bucketname = 'fasttext-predict-bucket'
files_names = ['ag_news_predict.txt',
               'dbpedia_predict.txt',
               'yelp_review_predict.txt',
               'sogou_predict.txt']


files_to_predict = list(map(lambda s: bucketname + '/' + s, files_names))


def map_fasttext_classification(key, data_stream):
    print('I am processing the object {}'.format(key))
    fasttext_model = fstTxt.load_model(ag_news_model)

    data = data_stream.read()
    result = list()
    for line in data.splitlines():
        label, prob = fasttext_model.predict(str(line))
        result.append((label[0], prob[0]))
    return result

def reduce_function(results, futures):
    all_result = list()
    for labels_list in results:
        all_result.extend(labels_list)
    run_statuses = [f.run_status for f in futures]
    invoke_statuses = [f.invoke_status for f in futures]
    return {"run_statuses": run_statuses, "invoke_statuses": invoke_statuses, "results": all_result}


chunk_size = 4 * 1024 ** 2  # 4MB
current_index = 0

repeats = 1
total_duration = 0
title = ""
graph_path = "../../InvocationsGraphsFiles/"

for index in range(repeats):
    executor = ExecutorWrap(0, title, graph_path)
    executor.set_location(exe_location)
    result_obj = executor.map_reduce_execution(map_fasttext_classification,
                                               files_to_predict[current_index],
                                               reduce_function,
                                               chunk_size,
                                               runtime="fasttext-exists-models")
    total_duration += executor.get_last_duration()

print("\nAVG Duration: " + str(total_duration / float(repeats)) + " Sec")
