
import csv
import json
import numpy as np
import pandas as pd
import scipy
from tabulate import tabulate
from tqdm import tqdm


class Evaluation(object):
    """
    - compute 4 metrics for logs
    - compute 4 metrics for elastic_search
    - print all the metrics (nicely if possible)
    - plot top k
    """

    def __init__(self, pir, metrics) -> None:
        self.pir = pir
        self.metrics = metrics

    def proceed(self) -> None:
        test_sample=np.infty
        self.sample_size=0
        self.elastic_hits=0  
        #Indexed as RR/topKrecall[A][B]:
        #   -A determines on which set of documnet the metric is computed
        #   -B determines on which ranking
        self.RR_computed={"log":{"log":0.0,"metrics":None},"ES":{"ES":0.0,"metrics":None}}
        self.top_K_recall_computed={"log":{"log":None,"metrics":None},"ES":{"ES":None,"metrics":None}}
        #tau/rDG[A], where A refers to the order it is compared with and the set of documents.
        self.tau_computed={"ES":None,"log":None}
        self.rDG_computed={"log":None,"ES":None}
        self.top_K_thresholds={"log":np.arange(1, 11),"ES":np.array([1,3,5,10,25,100,150])}
        if self.pir.dataset=="AOL":
            with open('datasets/AOL4PS/validation_data.csv') as f:
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
                    queryId=row[1]
                    queryText=self.pir.logManager.getQueryText(queryId)
                    user=row[0]
                    self.sample_size+=1
                    ES_docs,ES_scores=self.pir.clean_query(self.pir.query_es(queryText,250))
                    if ES_docs is None:
                        ES_docs,ES_scores=np.array(()), np.array(())
                    docs_to_rank, reference_scores = np.concatenate((np.array(log_rankings), ES_docs)), np.concatenate((np.zeros(len(log_rankings)), ES_scores))
                    scores=self.pir.reranker.evaluation_metrics_scores(queryId,user,docs_to_rank, reference_scores,session, metrics=self.metrics)
                    #Metrics computed on logs resulst only
                    logs_scores = scores[:len(log_rankings)]
                    rankings=self.compute_ranks_with_ties(logs_scores)
                    self.RR_computed["log"]["log"]=self.RR(rank_in_log,cumulative=self.RR_computed["log"]["log"])
                    self.RR_computed["log"]["metrics"]=self.RR(rankings[rank_in_log-1,:],cumulative=self.RR_computed["log"]["metrics"])
                    self.top_K_recall_computed["log"]["log"]=self.top_k_recall(rank_in_log,cumulative=self.top_K_recall_computed["log"]["log"],thresholds=self.top_K_thresholds["log"])
                    self.top_K_recall_computed["log"]["metrics"]=self.top_k_recall(rankings[rank_in_log-1,:],cumulative=self.top_K_recall_computed["log"]["metrics"],thresholds=self.top_K_thresholds["log"])
                    self.tau_computed["log"]=self.kendall_tau(np.arange(1,11),rankings,self.tau_computed["log"])
                    self.rDG_computed["log"]=self.rDG(rank_in_log,rankings[rank_in_log-1,:],self.rDG_computed["log"])
                    #Metrics computed on ES results
                    if(ES_scores is not None):
                        ES_ranking= self.compute_ranks_with_ties(ES_scores)
                        ind_ES=self.find_doc_index(ES_docs,relevant_doc)
                        if(ind_ES>0): #if the relevant doc is retrieved
                            rank_in_ES=ES_ranking[ind_ES]
                            self.elastic_hits+=1
                            #TODO: metrics considering the ES score and comparing reranking with initial ES
                            elastic_search_metrics = scores[len(log_rankings):]
                            rankings=self.compute_ranks_with_ties(elastic_search_metrics)
                            self.RR_computed["ES"]["ES"]=self.RR(rank_in_ES,cumulative=self.RR_computed["ES"]["ES"])
                            self.RR_computed["ES"]["metrics"]=self.RR(rankings[ind_ES,:],cumulative=self.RR_computed["ES"]["metrics"])
                            self.top_K_recall_computed["ES"]["ES"]=self.top_k_recall(rank_in_ES,cumulative=self.top_K_recall_computed["ES"]["ES"],thresholds=self.top_K_thresholds["ES"])
                            self.top_K_recall_computed["ES"]["metrics"]=self.top_k_recall(rankings[ind_ES,:],cumulative=self.top_K_recall_computed["ES"]["metrics"],thresholds=self.top_K_thresholds["ES"])
                            self.tau_computed["ES"]=self.kendall_tau(ES_ranking,rankings,self.tau_computed["ES"])
                            self.rDG_computed["ES"]=self.rDG(rank_in_ES,rankings[ind_ES,:],self.rDG_computed["ES"])
                    if(self.sample_size>=test_sample):
                        break

        top_k_columns = [f"top {k} recall" for k in self.top_K_thresholds["log"]]
        self.logs_results = pd.DataFrame(
            columns=["RR"] + top_k_columns + ["rDG", "tau"],
            index=self.metrics,
            data=np.hstack((
                self.RR_computed["log"]["metrics"].reshape(-1, 1), # array metrics
                self.top_K_recall_computed["log"]["metrics"], # matrix 2d (metrics, 4)
                self.rDG_computed["log"].reshape(-1, 1),# array metrics
                self.tau_computed["log"].reshape(-1, 1) # array metrics
            ))
        )
        self.logs_results.loc["reference"] = np.concatenate(([self.RR_computed["log"]["log"]], self.top_K_recall_computed["log"]["log"], [0, self.sample_size]))
        self.logs_results = self.logs_results / self.sample_size

        if self.elastic_hits<1:
            return
        top_k_columns = [f"top {k} recall" for k in self.top_K_thresholds["ES"]]
        self.elastic_results = pd.DataFrame(
            columns=["RR"] + top_k_columns + ["rDG", "tau"],
            index=self.metrics,
            data=np.hstack((
                self.RR_computed["ES"]["metrics"].reshape(-1, 1), # array metrics
                self.top_K_recall_computed["ES"]["metrics"], # matrix 2d (metrics, 4)
                self.rDG_computed["ES"].reshape(-1, 1),# array metrics
                self.tau_computed["ES"].reshape(-1, 1) # array metrics
            ))
        )
        self.elastic_results.loc["reference"] = np.concatenate(([self.RR_computed["ES"]["ES"]], self.top_K_recall_computed["ES"]["ES"], [0, self.sample_size ]))
        self.elastic_results = self.logs_results / self.sample_size

    def print(self) -> None:
        print(tabulate(self.logs_results, tablefmt="md", floatfmt=".4f", headers=self.logs_results.columns))

        if self.elastic_hits<1:
            print("\n\nElasticSearch never retrieved the relevant document, so no statistics can be computed.")
            return
        print(tabulate(self.elastic_results, tablefmt="md", floatfmt=".4f", headers=self.logs_results.columns))

    def store(self) -> None:
        data = {
            "sample size": self.sample_size,
            "logs results": json.loads(self.logs_results.to_json()),
        }
        if self.elastic_hits>0:
            data["elastic search hits"] = self.elastic_hits
            data["elastic search results"] = json.loads(self.elastic_results.to_json())
        
        with open("evaluation_results.json", "w") as file:
            file.write(json.dumps(data))

    def plot(self) -> None:
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
        # return scipy.stats.rankdata(-scores, axis=1, method="min")
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
        with np.errstate(divide='ignore',invalid='ignore'):
            result = (n_C-n_D)/(norm_factor*norm_factors)
            result[norm_factor*norm_factors == 0] = 0
        if cumulative is None:
            return result
        return cumulative+result
    
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
 