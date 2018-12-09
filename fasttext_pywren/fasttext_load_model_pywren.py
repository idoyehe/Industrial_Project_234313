import ibm_boto3
import os
import pywren_ibm_cloud as pywren
import sys
import yaml
import fastText as fstxt
import smart_open
import fasttext_pybind

config_filename = "~/.pywren_config"


try:
    config_path = os.path.join(os.path.expanduser(config_filename))
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
except Exception as e:
    print("can't open config file".format(e))
    sys.exit()
cos_endpoint = "s3.us-east.objectstorage.service.networklayer.com"
access_key = config['ibm_cos']['access_key']
secret_key = config['ibm_cos']['secret_key']


def my_function():
    cos_session = ibm_boto3.session.Session(aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name='us=east')
    storage_bucket = "fasttext-models"
    model_filepath = 's3://' + storage_bucket + '/all_reviews.bin'
    # fastText_model = fstxt.load_model(
    smart_open.smart_open(model_filepath, host=cos_endpoint, s3_session=cos_session)
    # return fastText_model.get_labels(include_freq=False)

my_function()
# pw = pywren.ibm_cf_executor()
# pw.call_async(my_function, fastText_model)
# print(pw.get_result())
