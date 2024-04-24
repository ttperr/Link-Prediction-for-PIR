
from elasticsearch import Elasticsearch

def establish_connection():
    return Elasticsearch('https://localhost:9200',
                           basic_auth=("elastic","oWOZ*md3bkO0dL3MNtrP"),
                           ssl_assert_fingerprint="01631128411294fce48a60a62920f30d01e311d60d3d864bbf4eee7c078a787c")