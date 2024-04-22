from tqdm import tqdm
from elasticsearch import Elasticsearch
import csv
import requests


def main():
    client = Elasticsearch('https://localhost:9200',
                           api_key='INSERT YOUR API KEY HERE',
                           ca_certs='~/certs/http_ca.crt')
    client.options(ignore_status=[400, 404]).indices.delete(index='aol4ps')
    with open('AOL4PS/doc.csv') as f:
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
    print('Data indexed successfully!')


if __name__ == '__main__':
    main()
