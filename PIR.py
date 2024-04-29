#! /usr/bin/env python3

import argparse
import sys
from indexer import index_in_elasticsearch
import Reranker
import connector
import csv
import math
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

    def query_es(self, search_text,n=None):
        if n == None:
            n = self.n_docs
        if self.dataset == "AOL":
            return self.client.search(index=self.index, query={"match": {"title": {"query": search_text, "fuzziness": "AUTO"}}}, size=n)
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
                    query_text=(queries[int(row[1][2:])+1]).split("\t")[0]
                    ES_rankings=self.clean_query(self.query_es(query_text,100))
                    
                    count+=1
                    #TODO: -rDG(rr(log docs only)+log)
                    #TODO: -tau(rr(log docs only)+log)
                    if(len(ES_rankings)>0):
                        #done: -topKrecall (ES,rr_noES,rr_ES)
                        #TODO: -MRR (ES, rr_noES,rr_ES)
                        #TODO: -tau(ES+rr_ES,ES+rr_noES)
                        #TODO: -rDG(ES+rr_ES,ES+rr_noES)
                        #TODO: -rDG(rr_ES+logs(EXTRA FILTERED))
                        ES_with_ties,ES_rank=self.aggregate_ties_finding_relevant(ES_rankings,relevant_doc)
                        if(ES_rank<0):
                            continue
                        relevant_retrieved_by_ES+=1
                        rr_ES_rankings=(self.reranker.rerank(query_text,ES_rankings,row[0]))
                        rr_ES_rankings.sort(key=lambda a: a[1], reverse=True)
                        rr_noES_rankings=(self.reranker.rerank(query_text,[(t[0],1.) for t in ES_rankings],row[0]))
                        rr_noES_rankings.sort(key=lambda a: a[1], reverse=True)
                        rr_ES_with_ties,rr_ES_rank=self.aggregate_ties_finding_relevant(rr_ES_rankings,relevant_doc)
                        rr_noES_with_ties,rr_noES_rank=self.aggregate_ties_finding_relevant(rr_noES_rankings,relevant_doc)
                        #recalls
                        self.top_k_recall(topKrecall["ES"],ES_rank)
                        self.top_k_recall(topKrecall["rr_ES"],rr_ES_rank)
                        self.top_k_recall(topKrecall["rr_noES"],rr_noES_rank)
                        #RR
                        RR["ES"]+=1/ES_rank
                        RR["rr_ES"]+=1/rr_ES_rank
                        RR["rr_noES"]+=1/rr_noES_rank
                        #rDG
                        rDG["rr_ES"]["ES"]+=self.rDG(rr_ES_rank,ES_rank)
                        rDG["rr_noES"]["ES"]+=self.rDG(rr_ES_rank,ES_rank)
                        #TODO rDG["rr_ES"]["log"]
                        #tau
                        tau["rr_ES"]["ES"]+=self.kendall_tau(rr_ES_with_ties,ES_with_ties)
                        tau["rr_noES"]["ES"]+=self.kendall_tau(rr_noES_with_ties,ES_with_ties)
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
