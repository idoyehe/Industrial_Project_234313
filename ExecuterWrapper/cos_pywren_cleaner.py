import sys
import yaml
import os
import ibm_boto3
from ibm_botocore.client import Config
from ibm_botocore.client import ClientError

PREFIX = 'pywren.jobs'
config_filename = "~/.pywren_config"
CONFIG = None
bucket_name = None

def init_cos_python_object():
    global CONFIG
    try:
        config_path = os.path.join(os.path.expanduser(config_filename))
        with open(config_path, 'r') as config_file:
            CONFIG = yaml.safe_load(config_file)
    except Exception as e:
        print("can't open config file".format(e))
        sys.exit()
    return ibm_boto3.resource("s3",
                              ibm_api_key_id=CONFIG['ibm_cos']['api_key'],
                              ibm_auth_endpoint='https://iam.ng.bluemix.net/oidc/token',
                              config=Config(signature_version="oauth"),
                              endpoint_url=CONFIG['ibm_cos']['endpoint']
                              )


def get_filenames_from_cos(cos, bucket_name, prefix):
    print("Retrieving items' names from bucket: {0}, prefix: {1}".format(bucket_name, prefix))
    result = []
    try:
        for data in cos.Bucket(bucket_name).objects.filter(Prefix=prefix):
            result.append(data.key)
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to delete item: {0}".format(e))
    return result


def delete_file_from_cos(cos, bucket_name, key):
    try:
        cos.Object(bucket_name, key).delete()
        print("File: {0} deleted!".format(key))
    except ClientError as be:
        print("CLIENT ERROR: {0}\n".format(be))
    except Exception as e:
        print("Unable to delete item: {0}".format(e))


def clean_pywren_cos():
    print('Deleting test files...')

    cos = init_cos_python_object()
    for key in get_filenames_from_cos(cos, CONFIG['pywren']['storage_bucket'], PREFIX):
        delete_file_from_cos(cos, CONFIG['pywren']['storage_bucket'], key)

    print("ALL DONE")


clean_pywren_cos()
