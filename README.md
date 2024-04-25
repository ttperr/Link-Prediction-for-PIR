# DD2477-SEIRS-KTH-project
Private repo of the DD2477 - Search Engines and Information Retrieval Systems project at KTH.

## Data
The data used in this project can be found [here](https://www.scidb.cn/en/detail?dataSetId=5246eba9ec8d4519aa4f0d8f9f092d4b#p4)

## Running
To run the code run from the terminal ``PIR.py``.
Flags:
+ ``-d \[AOL\]`` to change the dataset. If not specified, AOL4PS is used.
+ ``-l`` to index the chosen dataset into ElasticSearch. Should be done the first run only.
+ ``-n <integer>`` to specify the maximum number of files you are interested in. 


## Connector (TO DO: IN FINAL VERSION PUT THE CONNECTOR WITH FAKE PARAMS)
Add a connector.py with content:
```
from elasticsearch import Elasticsearch
def establish_connection():
    return Elasticsearch('https://localhost:9200',
                           basic_auth=("elastic","YOUR PW HERE"),
                           ssl_assert_fingerprint="YOUR SSL KEY HERE")
```
