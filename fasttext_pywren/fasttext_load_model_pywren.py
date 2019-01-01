from time import time
import fastText as fstTxt
import pywren_ibm_cloud as pywren

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


def map_fasttext_prediction(key, data_stream):
    print('I am processing the object {}'.format(key))
    fasttext_model = fstTxt.load_model(dbpedia_model)

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


chunk_size = 10 * 1024 ** 2  # 4MB

total_duration = 0
for index in range(1):
    start = time()
    pw = pywren.ibm_cf_executor(runtime="fasttext-exists-models")
    pw.map_reduce(map_fasttext_prediction, files_to_predict[1], reduce_function, chunk_size=chunk_size, reducer_wait_local=False)
    result_object = pw.get_result()
    if index == 0 and result_object['run_statuses'] and result_object['invoke_statuses']:
        pw.create_timeline_plots(dst="../InvocationsGraphsFiles/", name=files_names[1],
                                 run_statuses=result_object['run_statuses'], invoke_statuses=result_object['invoke_statuses'])
    end = time()
    print("index:", index, "time:", end - start)
    total_duration += end - start

print("Avg_time: ", total_duration / 5.0)
print("\n")