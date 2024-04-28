#! /usr/bin/env python3

import argparse
import sys
from indexer import index_in_elasticsearch
import Reranker
import connector
import csv
from tqdm import tqdm
from UI import Ui_Widget
from PySide6.QtWidgets import QApplication, QWidget
import datasets.AOL4PS.splitter as AOL_splitter


class PIR(object):
    def __init__(self, dataset, n_docs, validation=False):
        self.dataset = dataset
        self.n_docs = n_docs
        if (dataset == "AOL"):
            self.index = "aol4ps"
        else:
            raise NotImplementedError("Unknown dataset")

        self.client = connector.establish_connection()

        # In future we can add flags initializing different rerankers (graph, contents,...)
        self.reranker = Reranker.Reranker(dataset, validation)
        if validation:
            self.evaluate()

    def is_new_user(self, userId):
        return self.reranker.is_new_user(userId)

    def query(self, search_text, user):
        results = self.query_es(search_text)
        # print(results)
        clean_results = self.clean_query(results)
        print("ElasticSearch results (ordered)")
        print(clean_results)
        reranked_results = self.reranker.rerank(
            search_text, clean_results, user)
        print("Reranked results (ordered)")
        reranked_results.sort(key=lambda a: a[1], reverse=True)
        print(reranked_results)
        out_results = []
        if self.dataset == "AOL":
            # FASTER ways to do it, but since we have few results this is fine.
            for new_pos in range(len(reranked_results)):
                for old_pos in range(len(reranked_results)):
                    if (reranked_results[new_pos][0] == clean_results[old_pos][0]):
                        doc = results["hits"]["hits"][old_pos]
                        out_results.append(
                            (old_pos-new_pos, doc["_id"], doc["_source"]["url"], doc["_source"]["title"]))
                        break
            # return sorted list of (difference in rank, docID,bold_title,regular_description) for AOL
            return out_results
        return None

    def query_es(self, search_text):
        if self.dataset == "AOL":
            return self.client.search(index=self.index, query={"match": {"title": {"query": search_text, "fuzziness": "AUTO"}}}, size=self.n_docs)
        return None

    def clean_query(self, query_result):
        return ([(a["_id"], a["_score"]) for a in query_result["hits"]["hits"]])

    def register_click(self, doc_ids, user_id, doc_clicked_index, query_text):
        # doc_ids is a sorted list of [docid1, docid2], ordered from best to worst matching
        # user_id is the user id
        # doc_clicked_index is the index of the document clicked. From 0 (the best matching) to len(doc_ids)-1 (worst matching)
        # query_text is the query text.
        print("doc_ids", doc_ids)
        print("user_id", user_id)
        print("doc_clicked_index", doc_clicked_index)
        print("query_text", query_text)

        # TODO pass needed values to self.reranker to update the logs
    
    def evaluate(self):
        count=0
        relevant_retrieved_by_ES=0
        if self.dataset=="AOL":
            with open('datasets/AOL4PS/validation_data.csv') as f, open('datasets/AOL4PS/query.csv','r') as q:
                queries=q.readlines()
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Validation', unit='rows')
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    log_rankings=row[6].split()
                    rank_in_log=int(row[7])+1
                    relevant_doc=row[5]
                    query_text=(queries[int(row[1][2:])+1]).split("\t")[0]
                    ES_rankings=self.clean_query(self.query_es(query_text))
                    if(len(ES_rankings)>0):
                        ES_with_ties,rank_in_ES=self.aggregate_ties_finding_relevant(ES_rankings,relevant_doc)
                        #print(relevant_doc)
                        #print(ES_rankings)
                        #print(self.aggregate_ties_finding_relevant(ES_rankings,relevant_doc))
                        relevant_retrieved_by_ES+=1
                        count+=1
            print(count,"validation samples.")
            print(relevant_retrieved_by_ES,"times ES managed to retrieve the relevant doc.")
        else:
            raise NotImplementedError("Unknown dataset")
        pass

    def aggregate_ties_finding_relevant(self,clean_query_result,relevant_doc_id):
        result=[]
        tmp=[clean_query_result[0][0]]
        prev_score=clean_query_result[0][1]
        rank=-1
        if(clean_query_result[0][0]==relevant_doc_id):
            rank=1
        
        for t in range(1,len(clean_query_result)):
            if(clean_query_result[t][0]==relevant_doc_id):
                rank=len(result)+1
            if(clean_query_result[t][1]==prev_score):
                tmp.append(clean_query_result[t][0])
            else:
                prev_score=clean_query_result[t][1]
                result.append(tmp.copy())
                tmp=[clean_query_result[t][0]]
        result.append(tmp)
        return result,rank

class Widget(QWidget):
    def __init__(self, PIR, n, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self, PIR, n)


def main():
    parser = argparse.ArgumentParser(
        description="Personalized Information Retrieval project")
    parser.add_argument("-l", default=False, action='store_true',
                        help="Load data to ElasticSearch")
    parser.add_argument("-d", type=str, default="AOL",
                        help="Specify the dataset type. All parameters associated to dataset are hardcoded")
    parser.add_argument("-n", type=int, default=25,
                        help="Specify the maximum number of docs you want to return each search")
    parser.add_argument("-v", default=False, action="store_true",
                        help="Specify if you are performing validation")
    arguments = parser.parse_args()
    if (arguments.l):
        index_in_elasticsearch(arguments.d)
    if (arguments.v and arguments.d == "AOL"):
        AOL_splitter.validation_split()
    pir = PIR(arguments.d, arguments.n, arguments.v)
    app = QApplication(sys.argv)
    widget = Widget(pir, arguments.n)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
