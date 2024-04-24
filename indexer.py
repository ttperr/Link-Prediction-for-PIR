from tqdm import tqdm
import csv
import requests
import connector

#Loads data in the ElasticSearch engine
def index_in_ElasticSearch(dataset):

    client = connector.establish_connection()

    if dataset=="AOL":
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