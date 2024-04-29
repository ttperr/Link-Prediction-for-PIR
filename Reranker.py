import csv
import numpy as np
from tqdm import tqdm
from graph import Graph

class Reranker(object):

    #TODO: framework computation to get feature matrix

    def __init__(self,dataset,validation=False):
        self.dataset = dataset
        self.validation=validation
        self.ranking_ratio = 1
        if dataset=="AOL":
            #YOU SHOULD LOAD VARIABLES/USEFUL INFO FROM LOGS OR A FILE STORING THEM (IF YOU USE FILES, STORE THEM IN THE CORRECT FOLDER)
            #THE FOLDER with data and files needed to rerank SHOULD BE in  AOL4PS/ 
            self.user_document = Graph(0, 5, self.validation, dataset=self.dataset, name="user-document")
            
        else:
            raise NotImplementedError("Unknown dataset")

    def rerank(self,query_text,query_results,user):
        if len(query_results) == 0:
            return []
        queryID = self.getQueryID(query_text)
        # return self.PClick(queryID, query_results, user)
        return self.graph_metric(user, query_results, lambda u,d: 1/self.user_document.shortest_distance(u,d))
        return self.graph_metric(user, query_results, lambda u,d: 1/self.user_document.weighted_shortest_distance(u,d))
        return self.graph_metric(user, query_results, self.user_document.common_neighbors)
        return self.graph_metric(user, query_results, self.user_document.adamic_adar)
        return self.user_document_page_rank(user, query_results)

    def is_new_user(self,userID): #TODO
        #Must return true if the userId is new, false if it is known

        #JUST RANDOM FOR NOW
        if np.random.rand()>0.5:
            return True
        return False
    
    def getQueryID(self, query_text):
        """parse query.csv and return matching query id"""
        maxId = 0
        if self.dataset=="AOL":
            with open('datasets/AOL4PS/query.csv') as f:
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Parsing queries', unit='rows')
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    if row[0] == query_text:
                        return row[1]
                    idNumber = int(row[1][2:])
                    if idNumber > maxId:
                        maxId = idNumber
        else:
            raise NotImplementedError("Unknown dataset")
        return f"q-{maxId+1}"


    def PClick(self, queryID, query_results, userID):
        """Hello world metric, defined in https://dl.acm.org/doi/10.1145/1242572.1242651"""
        ranking_ratio = 1
        potentialMatches = 0
        matches = {}
        beta = 0.5
        res=[]
        if self.dataset=="AOL":
            with open('datasets/AOL4PS/data.csv') as f:
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Parsing logs', unit='rows')
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    if row[1] == queryID and row[0] == userID:
                        potentialMatches += 1
                        matches[row[5]] =  (matches[row[5]] + 1 if row[5] in matches else 1)
        else:
            raise NotImplementedError("Unknown dataset")
        for resultID ,score in query_results:
            newScore = score * (1 - ranking_ratio) + (matches[resultID]/(potentialMatches+beta) if resultID in matches else 0) * ranking_ratio
            res.append((resultID, newScore))
        return res
    
    def graph_metric(self, user, documents, metric):
        return [self.combine_score(document, lambda d: metric(user, d)) for document in documents]

    def user_document_page_rank(self, user, documents):
        pagerank_scores = self.user_document.rooted_page_rank(user, documents)
        return [self.combine_score(doc_and_score, pagerank) for doc_and_score, pagerank in zip(documents, pagerank_scores)]

    def combine_score(self, doc_and_score, metric):
        document, score = doc_and_score
        if callable(metric):
            return (document, score * (1 - self.ranking_ratio) + metric(document) * self.ranking_ratio)
        return (document, score * (1 - self.ranking_ratio) + metric * self.ranking_ratio)
    
    def prop_flow_ranking(self, user, documents):
        ranking_ratio = 0.5  
        res = []
        for document, score in documents:
            metric = self.user_document.prop_flow(user, document, max_length=5)
            new_score = score * (1 - ranking_ratio) + metric * ranking_ratio
            res.append((document, new_score))
        return res  

    def graph_features(self, query, user, documents, session=None):
        tmp = self.ranking_ratio
        self.ranking_ratio = 1
        res = np.matrix(
            self.graph_metric(user, documents, lambda u,d: self.user_document.degree(u)),
            self.graph_metric(user, documents, lambda u,d: self.user_document.degree(d)),
            self.graph_metric(user, documents, lambda u,d: 1/self.user_document.shortest_distance(u, d)),
            self.graph_metric(user, documents, lambda u,d: 1/self.user_document.weighted_shortest_distance(u, d)),
            self.graph_metric(user, documents, self.user_document.common_neighbors),
            self.graph_metric(user, documents, self.user_document.adamic_adar),
            self.user_document_page_rank(user, documents),
            self.graph_metric(session, documents, lambda u,d: self.session_document.degree(u)),
            self.graph_metric(session, documents, lambda u,d: self.session_document.degree(d)),
            self.graph_metric(session, documents, lambda u,d: 1/self.session_document.shortest_distance(u, d)),
            self.graph_metric(session, documents, lambda u,d: 1/self.session_document.weighted_shortest_distance(u, d)),
            self.graph_metric(session, documents, self.session_document.common_neighbors),
            self.graph_metric(session, documents, self.session_document.adamic_adar),
            self.session_document_page_rank(session, documents),
            self.graph_metric(user, [(query, 0)], lambda u,d: self.user_query.degree(u)),
            self.graph_metric(user, [(query, 0)], lambda u,d: self.user_query.degree(d)),
            self.graph_metric(user, [(query, 0)], lambda u,d: 1/self.user_query.shortest_distance(u, d)),
            self.graph_metric(user, [(query, 0)], lambda u,d: 1/self.user_query.weighted_shortest_distance(u, d)),
            self.graph_metric(user, [(query, 0)], self.user_query.common_neighbors),
            self.graph_metric(user, [(query, 0)], self.user_query.adamic_adar),
            self.graph_metric(session, [(query, 0)], lambda u,d: self.session_query.degree(u)),
            self.graph_metric(session, [(query, 0)], lambda u,d: self.session_query.degree(d)),
            self.graph_metric(session, [(query, 0)], lambda u,d: 1/self.session_query.shortest_distance(u, d)),
            self.graph_metric(session, [(query, 0)], lambda u,d: 1/self.session_query.weighted_shortest_distance(u, d)),
            self.graph_metric(session, [(query, 0)], self.session_query.common_neighbors),
            self.graph_metric(session, [(query, 0)], self.session_query.adamic_adar),
            self.graph_metric(query, documents, lambda u,d: self.query_document.degree(u)),
            self.graph_metric(query, documents, lambda u,d: self.query_document.degree(d)),
            self.graph_metric(query, documents, lambda u,d: 1/self.query_document.shortest_distance(u, d)),
            self.graph_metric(query, documents, lambda u,d: 1/self.query_document.weighted_shortest_distance(u, d)),
            self.graph_metric(query, documents, self.query_document.common_neighbors),
            self.graph_metric(query, documents, self.query_document.adamic_adar)
        )
        self.ranking_ratio = tmp
        return res
    
          


    

