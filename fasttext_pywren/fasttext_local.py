import fastText as fstTxt
from time import time

ag_news_model = "./ag_news/ag_news.ftz"  # env arrange
amazon_review_model = "./amazon_review_polarity/amazon_review_polarity.ftz"  # env arrange
dbpedia_model = "./dbpedia/dbpedia.ftz"  # env arrange
sogou_news_model = "./sogou_news/sogou_news.ftz"  # env arrange
yelp_review_model = "./yelp_review_full/yelp_review_full.ftz"  # env arrange

bucketname = 'fasttext-predict-bucket'

files_names = {"ag_news": './ag_news/ag_news_predict.txt',  # ~ 4 sec
               "dbpedia": './dbpedia/dbpedia_predict.txt',  # ~ 23 sec
               "sogou_news": './sogou_news/sogou_predict.txt',  # ~ 140 sec
               "yelp": './yelp_review_full/yelp_review_predict.txt'}  # ~  50 sec


def ag_news_predictions():
    fasttext_model = fstTxt.load_model(ag_news_model)
    input_file = open(files_names['ag_news'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        result.append(fasttext_model.predict(line))
    return result


def dbpedia_predictions():
    fasttext_model = fstTxt.load_model(dbpedia_model)
    input_file = open(files_names['dbpedia'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        result.append(fasttext_model.predict(line))
    return result


def yelp_predictions():
    fasttext_model = fstTxt.load_model(yelp_review_model)
    input_file = open(files_names['yelp'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        result.append(fasttext_model.predict(line))
    return result


def sogou_news_predictions():
    fasttext_model = fstTxt.load_model(sogou_news_model)
    input_file = open(files_names['sogou_news'], 'r').read()

    result = list()
    for line in input_file.splitlines():
        result.append(fasttext_model.predict(line))
    return result


start = time()

ag_news_predictions()
# dbpedia_predictions()
# yelp_predictions()
# sogou_news_predictions()

end = time()
duration = end - start
print("\nDuration: " + str(duration) + " Sec")
