# Link Prediction for PIR
Private repo of the DD2477 - Search Engines and Information Retrieval Systems project at KTH.

> Done by Titouan Mazier, Lorenzo Sibille, Tristan Perrot & Hani Anouar Bourrous

## Data
The data used in this project can be found and downloaded [here](https://www.scidb.cn/en/detail?dataSetId=5246eba9ec8d4519aa4f0d8f9f092d4b#p4).

## Connect to ElasticSearch
Update the values in the file ```connector.py```, inserting your password and other fields needed to setup a connection. Multiple options are possible, see ElasticSearch documentation [here](https://www.elastic.co/guide/en/elasticsearch/client/python-api/current/connecting.html).


## Running
To run the code run from the terminal you need to launch an docker instance of [**Elastic Search**](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html) and run **``PIR.py``** file. \
Flags:
+ ``-d \[AOL\]`` to change the dataset. If not specified, AOL4PS is used.
+ ``-l`` to index the chosen dataset into ElasticSearch. Should be done the **first run only**.
+ ``-n <integer>`` to specify the maximum number of files you are interested in.
+ ``-v`` to run the code in evaluation mode. It will compute and print metrics about the reranking process, without running the GUI.
