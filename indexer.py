import csv
import sys

import requests
from tqdm import tqdm

import connector

csv.field_size_limit(sys.maxsize)


def index_in_elasticsearch(dataset):
    """
    Indexes data in ElasticSearch.

    Args:
        dataset (str): The dataset to be indexed. Currently supports "AOL".

    Raises:
        NotImplementedError: If an unknown dataset is provided.

    Returns:
        None
    """

    client = connector.establish_connection()

    if dataset == "AOL":
        client.options(ignore_status=[400, 404]).indices.delete(index='aol4ps')
        with open('datasets/AOL4PS/doc.csv') as f:
            reader = csv.reader(f, delimiter='\t')
            firstRow = True
            pbar = tqdm(reader, desc='Indexing data', unit='rows')
            for row in pbar:
                if firstRow:
                    firstRow = False
                    continue
                doc = {
                    'url': row[0],
                    'title': row[2]
                }
                client.index(index='aol4ps', body=doc, id=row[1])
    else:
        raise NotImplementedError("Unknown dataset")

    print('Data indexed successfully!')
