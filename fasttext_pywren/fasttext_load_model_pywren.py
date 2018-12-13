import fastText as fstxt
import pywren_ibm_cloud as pywren


model_path = "/fasttext/models/yelp_review_full.ftz"

def my_function(x):
    fasttext_model = fstxt.load_model(model_path)
    return fasttext_model.predict("IDO is Best men"), fasttext_model.predict("IDO is Worst men")


pw = pywren.ibm_cf_executor(runtime="fasttext-exists-models")
pw.call_async(my_function, 3)
print(pw.get_result())
