
from elasticsearch import Elasticsearch

def establish_connection():
    return Elasticsearch("YOUR PARAMS HERE")