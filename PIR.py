#! /usr/bin/env python3

import argparse
import sys
from indexer import index_in_elasticsearch
import Reranker
import connector
import csv
import math
from tqdm import tqdm
import numpy as np
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
        docs,scores = self.clean_query(results)
        print("ElasticSearch results (ordered)")
        print(docs,scores)
        if docs is None:
            return []
        reranked_scores = self.reranker.rerank(
            search_text, user,docs,scores)
        print("Reranked results (index corrresponds to previous docs)")
        print(reranked_scores)
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
        print("doc_ids", doc_ids)
        print("user_id", user_id)
        print("doc_clicked_index", doc_clicked_index)
        print("query_text", query_text)

        # TODO pass needed values to self.reranker to update the logs
    
    def evaluate(self):
        test_sample=50 #TODO remove
        count=0
        relevant_retrieved_by_ES=0
        #Using as key: -ES: original results by ES; -rr_ES: reranked considering ES; -rr_noES: reranked not considering ES weights; -log: log data
        RR={"rr_ES":0.,"rr_noES":0.0,"ES":0.0}
        tau={"rr_ES":{"ES":0.},"rr_noES":{"ES":0.,"log":0}}
        rDG={"rr_ES":{"log":0,"ES":0},"rr_noES":{"log":0.,"ES":0.}}
        topKrecall={"rr_ES":self.top_k_recall(),"rr_noES":self.top_k_recall(),"ES":self.top_k_recall()}
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
                    session=row[3]
                    query_text=(queries[int(row[1][2:])+1]).split("\t")[0]
                    user=row[0]
                    count+=1
                    #TODO: metrics considering the log only, and comparing rerank weight only (no ES) with logs
                    self.reranker.evaluation_metrics_scores(query_text,user,np.array(log_rankings),np.ones(len(log_rankings)),session)
                    
                    ES_docs,ES_scores=self.clean_query(self.query_es(query_text,100))
                    if(ES_scores is not None):
                        
                        if(relevant_doc not in ES_docs): #if the relevant doc is not retrieved
                            continue
                        relevant_retrieved_by_ES+=1
                        #TODO: metrics considering the ES score and comparing reranking with initial ES
                    if(count>=test_sample):
                        break
                        
            print(count,"validation samples.")
            print("Over the ",relevant_retrieved_by_ES,"(","{:.2f}".format(relevant_retrieved_by_ES/count),") times ES managed to retrieve the relevant doc, we have:")
            self.top_k_recall(print_t=True)
            print(topKrecall)
            print(rDG)
        else:
            raise NotImplementedError("Unknown dataset")
        pass

    def compute_ranks_with_ties(self,scores):
        '''
        Gives ranking based on scores

        Args:
            scores: a np.array of floats, containing the scores. The indexes are the same as doc ids. May be unsorted. 
                    Can be a matrix where each row is a metric, and each column a document.

        Returns:
            ranks: a np.array containing the rank for each score. Has the same dimensions as scores.
        '''
        
        if(scores.n_dim<2):
            scores.reshape(-1,1)
        ranks=np.zeros(scores.shape,dtype=np.int32)
        arg_ranks=np.argsort(scores,axis=1)
        for i in range(arg_ranks):
            score=-1
            r=1
            t=1
            for j in arg_ranks[i,:]:
                if scores[i,j]!=score:
                    t=r
                ranks[i,j]=t
                r+=1
        return ranks

        
    def kendall_tau(self,r1_with_ties,r2_with_ties):
        #TODO
        return 0.
    def rDG(self,r1,r2):
        if r1==r2:
            return 0.
        return (r2-r1)/(abs(r1-r2))*math.log(1+abs(r1-r2))/math.log(1+min(r1,r2))
    def top_k_recall(self,cumulative=None,r=None,print_t=False):
        thresholds=[1,3,5,10,25,100]
        if(print_t):
            print(thresholds)
            return
        if cumulative is None:
            return [0 for i in thresholds]
        else:
            for i,t in enumerate(thresholds):
                if r<=t:
                    cumulative[i]+=1
        

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
