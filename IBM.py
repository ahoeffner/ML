# Do Magic to make python import work

import pandas as pd
import ibm_boto3
from botocore.client import Config

def __iter__(self): return 0

# @hidden_cell
# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

cos_client = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='05wZ7sumrhQBRZMQ92yYbSo35bPTDfg9QlUGKQQSlsD_',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/identity/token",
    config=Config(signature_version='oauth'),
    endpoint_url='https://s3.direct.eu-de.cloud-object-storage.appdomain.cloud')

bucket = 'prediction-donotdelete-pr-gh5auxzpnvsvgs'
object_key = 'Predictions__da__ssvtungt62.py'

# load data of type "text/x-script.python" into a botocore.response.StreamingBody object.
# Please read the documentation of ibm_boto3 and pandas to learn more about the possibilities to load the data.
# ibm_boto3 documentation: https://ibm.github.io/ibm-cos-sdk-python/
# pandas documentation: http://pandas.pydata.org/

streaming_body_1 = cos_client.get_object(Bucket=bucket, Key=object_key)['Body']

f = open('WXPredictions.py', 'wb')
f.write(streaming_body_1.read())
f.close()

print("Predictions.py loaded")




# Next import new loaded module and modules for loading data

import pandas as pd
from WXPredictions import Predictions
from ibm_watson_studio_lib import access_project_or_space

access_token = "p-2+hI7h6fZGf0WYAZl3ZyfRjQ==;mhrdDJzybWxxRZ0oSOfOBA==:0qBsxCcygc9vu/lWjn9qUckY8K8KCodYw4avm2gUX5HLhvRio6Vs9W2VK01fuyjqsrIgSdH0jfzZ/fJNkaJ0aidJ7SPVRlAs1w=="


# Load data from IBM Cloud Object Storage

lib = access_project_or_space({'token':access_token})
print(lib)

file = lib.load_data("attrition.csv")
file.seek(0)

df = pd.read_csv(file)
predict = Predictions(None)
predict.predict(df)