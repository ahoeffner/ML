import pandas as pd
from ibm_watson_studio_lib import access_project_or_space

access_token = "p-2+hI7h6fZGf0WYAZl3ZyfRjQ==;mhrdDJzybWxxRZ0oSOfOBA==:0qBsxCcygc9vu/lWjn9qUckY8K8KCodYw4avm2gUX5HLhvRio6Vs9W2VK01fuyjqsrIgSdH0jfzZ/fJNkaJ0aidJ7SPVRlAs1w=="

# Load data from IBM Cloud Object Storage
lib = access_project_or_space({'token':access_token})

file = lib.load_data("attrition.csv")
file.seek(0)

df = pd.read_csv(file)

predictions = Predictions()
predictions.train(data=df)