import fastText as fstxt
import pywren_ibm_cloud as pywren


model_path = "/fasttext/models/ag_news.ftz"

def my_function(x):
    fasttext_model = fstxt.load_model(model_path)
    return fasttext_model.predict("IDO is Best men"), fasttext_model.predict("IDO is Worst men")


pw = pywren.ibm_cf_executor(runtime="fasttext-ag-news")
pw.call_async(my_function, 3)
print(pw.get_result())
