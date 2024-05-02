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
        #Indexed as RR/topKrecall[A][B]:
        #   -A determines on which set of documnet the metric is computed
        #   -B determines on which ranking
        RR={"log":{"log":0.0,"metrics":None},"ES":{"ES":0.0,"metrics":None}}
        topKrecall={"log":{"log":None,"metrics":None},"ES":{"ES":None,"metrics":None}}
        #tau/rDG[A], where A refers to the order it is compared with and the set of documents.
        tau={"ES":None,"log":None}
        rDG={"log":None,"ES":None}
        topKthresholds={"log":np.array([1,3,5,10]),"ES":np.array([1,3,5,10,25,100,150])}
        if self.dataset=="AOL":
            with open('datasets/AOL4PS/validation_data.csv') as f, open('datasets/AOL4PS/query.csv','r') as q:
                queries=q.readlines()
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Validation', unit='rows')
                metrics=[]
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
                    #Metrics computed on logs resulst only
                    scores,metrics=self.reranker.evaluation_metrics_scores(query_text,user,np.array(log_rankings),np.ones(len(log_rankings)),session)
                    rankings=self.compute_ranks_with_ties(scores)
                    RR["log"]["log"]=self.RR(rank_in_log,cumulative=RR["log"]["log"])
                    RR["log"]["metrics"]=self.RR(rankings[rank_in_log-1,:],cumulative=RR["log"]["metrics"])
                    topKrecall["log"]["log"]=self.top_k_recall(rank_in_log,cumulative=topKrecall["log"]["log"],thresholds=topKthresholds["log"])
                    topKrecall["log"]["metrics"]=self.top_k_recall(rankings[rank_in_log-1,:],cumulative=topKrecall["log"]["metrics"],thresholds=topKthresholds["log"])
                    tau["log"]=self.kendall_tau(np.arange(1,11),rankings,tau["log"])
                    rDG["log"]=self.rDG(rank_in_log,rankings[rank_in_log-1,:],rDG["log"])
                    #TODO taus and rDG
                    #Metrics computed on ES results
                    ES_docs,ES_scores=self.clean_query(self.query_es(query_text,250))
                    if(ES_scores is not None):
                        ES_ranking= self.compute_ranks_with_ties(ES_scores)
                        ind_ES=self.find_doc_index(ES_docs,relevant_doc)
                        if(ind_ES>0): #if the relevant doc is retrieved
                            rank_in_ES=ES_ranking[ind_ES]
                            relevant_retrieved_by_ES+=1
                            #TODO: metrics considering the ES score and comparing reranking with initial ES
                            scores,_=self.reranker.evaluation_metrics_scores(query_text,user,ES_docs,ES_scores,session)
                            rankings=self.compute_ranks_with_ties(scores)
                            RR["ES"]["ES"]=self.RR(rank_in_ES,cumulative=RR["ES"]["ES"])
                            RR["ES"]["metrics"]=self.RR(rankings[ind_ES,:],cumulative=RR["ES"]["metrics"])
                            topKrecall["ES"]["ES"]=self.top_k_recall(rank_in_ES,cumulative=topKrecall["ES"]["ES"],thresholds=topKthresholds["ES"])
                            topKrecall["ES"]["metrics"]=self.top_k_recall(rankings[ind_ES,:],cumulative=topKrecall["ES"]["metrics"],thresholds=topKthresholds["ES"])
                            tau["ES"]=self.kendall_tau(ES_ranking,rankings,tau["ES"])
                            rDG["ES"]=self.rDG(rank_in_ES,rankings[ind_ES,:],rDG["ES"])
                    if(count>=test_sample):
                        break
            
                print("\n\nMetrics computed on",count,"validation samples.")
                print("Considering only the log documents, without any intervention of ES, we have:")
                print("Top k recall (mean):")
                print("\tk=\t",topKthresholds["log"])
                print("\t log",topKrecall["log"]["log"]/count)
                for i,met in enumerate(metrics):
                    print(met,topKrecall["log"]["metrics"][i,:]/count)
                print("\nMean reciprocal rank:")
                print("\t log",RR["log"]["log"]/count)
                for i,met in enumerate(metrics):
                    print("\t",met,RR["log"]["metrics"][i]/count)
                print("\nrDG (mean):")
                for i,met in enumerate(metrics):
                    print("\t",met,rDG["log"][i]/count)
                print("\nTau (mean):")
                for i,met in enumerate(metrics):
                    print("\t",met,tau["log"][i]/count)

                print("\n\nMetrics computed on",relevant_retrieved_by_ES,"validation samples in which ElasticSearch returned the relevant document (",(relevant_retrieved_by_ES/count),"% ).")
                print("Top k recall (mean):")
                print("\tk=\t",topKthresholds["ES"])
                print("\t ES",topKrecall["ES"]["ES"]/relevant_retrieved_by_ES)
                for i,met in enumerate(metrics):
                    print(met,topKrecall["ES"]["metrics"][i,:]/relevant_retrieved_by_ES)
                print("\nMean reciprocal rank:")
                print("\t ES",RR["ES"]["ES"]/relevant_retrieved_by_ES)
                for i,met in enumerate(metrics):
                    print("\t",met,RR["ES"]["metrics"][i]/relevant_retrieved_by_ES)
                print("\nrDG (mean):")
                for i,met in enumerate(metrics):
                    print("\t",met,rDG["ES"][i]/relevant_retrieved_by_ES)
                print("\nTau (mean):")
                for i,met in enumerate(metrics):
                    print("\t",met,tau["ES"][i]/count)
                #print("Over the ",relevant_retrieved_by_ES,"(","{:.2f}".format(relevant_retrieved_by_ES/count),") times ES managed to retrieve the relevant doc, we have:")
        else:
            raise NotImplementedError("Unknown dataset")
        pass

    def compute_ranks_with_ties(self,scores):
        '''
        Gives ranking based on scores

        Args:
            scores: a np.array of floats, containing the scores. The indexes are the same as doc ids. May be unsorted. 
                    Can be a matrix where each row is a document, and each column a metric.

        Returns:
            ranks: a np.array containing the rank for each score. Has the same dimensions as scores.
        '''
        if(scores.ndim<2):
            scores=scores.reshape(-1,1)
        ranks=np.zeros(scores.shape,dtype=np.int32)
        arg_ranks=np.argsort(-scores,axis=0)
        for i in range(scores.shape[1]):
            score=-1
            r=1
            t=1
            for k in range(scores.shape[0]):
                j=arg_ranks[k,i]
                if scores[j,i]!=score:
                    t=r
                    score=scores[j,i]
                ranks[j,i]=t
                r+=1
        return ranks
    def find_doc_index(self,doc_list,rel_doc):
        '''
        Find the index of the relevant doc in a list of doc ids. If not present, return -1.

        Args:
            doc_list: a list of strings, containing the doc ids.
            rel_doc: a string representing the id of the relevant doc

        Returns:
            ind: an int representing the index of rel_doc in doc_list. -1 if not present.
        '''
        
        ind=np.nonzero(doc_list==rel_doc)[0]
        if ind.size==0:
            return -1
        return ind[0]
        
    def kendall_tau(self,ranking,rankings,cumulative=None):
        ranking=ranking.reshape(-1,1)
        if rankings.ndim<2:
            rankings=rankings.reshape(-1,1)
        '''
        n_c=np.zeros(rankings.shape[1])
        n_d=np.zeros(rankings.shape[1])
        for d in range(ranking.shape[0]-1):
            partial= (ranking[d+1:,:]-ranking[d,:]) * (rankings[d+1:,:]-rankings[d,:])
            n_c+=(partial >0).sum(axis=0)
            n_d+=(partial<0).sum(axis=0)
        '''
        partial=(np.expand_dims(ranking,0)-np.expand_dims(ranking,1))*(np.expand_dims(rankings,0)-np.expand_dims(rankings,1))
        n_C=(partial>0).sum(axis=(0,1))
        n_D=(partial<0).sum(axis=(0,1))
        _,counts=np.unique(ranking,return_counts=True)
        
        n_0=rankings.shape[0]*(rankings.shape[0]-1)/2

        norm_factor=np.sqrt(n_0-(counts*(counts-1)).sum()/2)
        norm_factors=np.zeros(rankings.shape[1])
        for i in range(rankings.shape[1]):  
            _,counts=np.unique(rankings[:,i],return_counts=True)
            norm_factors[i]=np.sqrt(n_0-(counts*(counts-1)).sum()/2)
        return (n_C-n_D)/(norm_factor*norm_factors)
    def rDG(self,rank,ranks,cumulative=None):
        '''
        Gives ranking based on scores

        Args:
            rank: an int representing the rank we are comparing all the others with.
            ranks: a np.array of ints, representing the rank of the relevant doc from various metrics, which are compared to rank.
            cumulative: the cumulative current rDG

        Returns:
            rdg: a np.array of floats of the same dimension of ranks, containing the cumulative+rDG of ranks wrt rank.
        '''
        diff=ranks-rank
        if cumulative is None:
            return np.divide(-diff*np.log2(1+np.abs(diff)), np.abs(diff)*np.log2(1+np.minimum(ranks,rank)), out=np.zeros_like(ranks,dtype=np.float32), where=diff!=0)
        return cumulative+np.divide(-diff*np.log2(1+np.abs(diff)), np.abs(diff)*np.log2(1+np.minimum(ranks,rank)), out=np.zeros_like(ranks,dtype=np.float32), where=diff!=0)

    def top_k_recall(self,ranks,thresholds=np.array([1,3,5,10]),cumulative=None):
        '''
        Gives ranking based on scores

        Args:
            ranks: a np.array of ints, representing the rank of the relevant doc from various metrics, which are compared to rank.
            cumulative: the cumulative current top_k_recalls

        Returns:
            top_k_recall: a bidimensional np.array of floats of the having ranks * threshold dimension, containing the cumulative+top_K_recall of ranks.
        '''
        if isinstance(ranks,np.ndarray):
            thresholds=thresholds.reshape(1,-1)
            ranks=ranks.reshape(-1,1)
            out=np.zeros((ranks.shape[0],thresholds.shape[1]))
        else:
            out=np.zeros(thresholds.shape[0])
        out[thresholds>=ranks]=1
        if cumulative is None:
            return out
        return cumulative+out
    
    def RR(self,ranks,cumulative=None):
        if cumulative is None:
            return 1/ranks
        else:
            return cumulative+1/ranks

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
    app = QApplication(sys.argv)
    widget = Widget(pir, arguments.n)
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
