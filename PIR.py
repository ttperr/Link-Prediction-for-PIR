#! /usr/bin/env python3

import argparse
import sys
from indexer import index_in_elasticsearch
import Reranker
import connector
import LogManager
import csv
import math
from tqdm import tqdm
import numpy as np
from UI import Ui_Widget
from PySide6.QtWidgets import QApplication, QWidget
import datasets.AOL4PS.splitter as AOL_splitter

from evaluation import Evaluation

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
        self.logManager= LogManager.LogManager(dataset)
        if validation:
            self.evaluation = Evaluation(self, [
                # "user_document-shortest_distance",
                # "user_document-weighted_shortest_distance",
                # "user_document-common_neighbors",
                # "user_document-adamic_adar",
                # "user_document-page_rank",
                "user_document-prop_flow",
            ])
            self.evaluation.proceed()
            self.evaluation.print()

    def is_new_user(self, userId):
        return self.reranker.is_new_user(userId)

    def query(self, search_text, user):
        results = self.query_es(search_text)
        # print(results)
        docs,scores = self.clean_query(results)
        print("ElasticSearch results (ordered, top 10)")
        print(docs[:min(10,scores.shape[0]-1)],scores[:min(10,scores.shape[0]-1)])
        if docs is None:
            return []
        queryId,_=self.logManager.getQueryID(search_text)
        reranked_scores = self.reranker.rerank(
            queryId, user,docs,scores)
        print("Reranked results (index corrresponds to previous docs)")
        print(reranked_scores[:min(10,scores.shape[0]-1)])
        reranked_docs=docs[np.argsort(-reranked_scores)]
        out_results = []
        if self.dataset == "AOL":
            # FASTER ways to do it, but since we have few results this is fine.
            for new_pos in range(len(docs)):
                for old_pos in range(len(docs)):
                    if (reranked_docs[new_pos] == docs[old_pos]):
                        doc = results["hits"]["hits"][old_pos]
                        out_results.append(
                            (old_pos-new_pos, doc["_id"], doc["_source"]["url"], doc["_source"]["title"]))
                        break
            # return sorted list of (difference in rank, docID,bold_title,regular_description) for AOL
            return out_results
        return None

    def query_es(self, search_text,n=None):
        if n is None:
            n = self.n_docs
        if self.dataset == "AOL":
            return self.client.search(index=self.index, query={"match": {"title": {"query": search_text, "fuzziness": "AUTO"}}}, size=n)
        return None

    def clean_query(self, query_result):
        """
        Cleans the query results returned by elastic search.

        Args:
            query_result: the result of the query from elastic search 

        Returns:
            None,None: if no documents are returned by ElasticSearch
            docs,scores: where doc is a np.array of strings and scores is a np.array of scores, with matching indexes.
        """
        if len(query_result["hits"]["hits"])==0:
            return None,None
        return np.array([a["_id"] for a in query_result["hits"]["hits"]]), np.array([a["_score"] for a in query_result["hits"]["hits"]]) 

    def register_click(self, doc_ids, user_id, doc_clicked_index, query_text):
        # doc_ids is a sorted list of [docid1, docid2], ordered from best to worst matching
        # user_id is the user id
        # doc_clicked_index is the index of the document clicked. From 0 (the best matching) to len(doc_ids)-1 (worst matching)
        # query_text is the query text.
        queryId,session,isUserOk=self.logManager.register_log(doc_ids, user_id, doc_clicked_index, query_text)
        if isUserOk:
            self.reranker.updateGraphFromClicks(user_id,session,queryId,doc_ids[doc_clicked_index])
    

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
    parser.add_argument("-n", type=int, default=100,
                        help="Specify the maximum number of docs you want to return each search")
    parser.add_argument("-v", default=False, action="store_true",
                        help="Specify if you are performing validation")
    arguments = parser.parse_args()
    if (arguments.l):
        index_in_elasticsearch(arguments.d)
    if (arguments.v and arguments.d == "AOL"):
        AOL_splitter.validation_split()
        pir = PIR(arguments.d, arguments.n, arguments.v)
        return
    pir = PIR(arguments.d, arguments.n, arguments.v)
    app = QApplication(sys.argv)
    widget = Widget(pir, arguments.n)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
