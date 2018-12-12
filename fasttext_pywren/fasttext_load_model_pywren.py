import fastText as fstxt


model_path = "./amazon_real_reviews_model/database/all_reviews_quantize.bin"

def my_function():
    fasttext_model = fstxt.load_model(model_path)
    print(fasttext_model.predict("IDO is Best men"))
    print(fasttext_model.predict("IDO is Worst men"))
    print(fasttext_model.predict("IDO"))
    print(fasttext_model.predict("IDO"))
    print(fasttext_model.predict("IDO"))
    print(fasttext_model.predict("IDO"))
    print(fasttext_model.predict("IDO"))


my_function()
