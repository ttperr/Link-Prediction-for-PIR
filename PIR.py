#! /usr/bin/env python3

import argparse
import sys
from indexer import index_in_elasticsearch
import Reranker
import connector

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
                    pass  # TODO
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
            return self.client.search(index=self.index, query={"match": {"title": search_text}}, size=self.n_docs)
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
