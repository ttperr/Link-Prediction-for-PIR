import csv
import time
import numpy as np
from tqdm import tqdm
from graph import Graph

class Reranker(object):

    def __init__(self,dataset,validation=False):
        self.dataset = dataset
        self.validation=validation
        self.ranking_ratio = 1
        if dataset=="AOL":
            #YOU SHOULD LOAD VARIABLES/USEFUL INFO FROM LOGS OR A FILE STORING THEM (IF YOU USE FILES, STORE THEM IN THE CORRECT FOLDER)
            #THE FOLDER with data and files needed to rerank SHOULD BE in  AOL4PS/ 
            self.user_document = Graph(0, 5, self.validation, dataset=self.dataset, name="user-document")
            # self.query_document = Graph(1, 5, self.validation, dataset=self.dataset, name="query-document")
            # self.query_user= Graph(1, 0, self.validation, dataset=self.dataset, name="query-user")
        else:
            raise NotImplementedError("Unknown dataset")

    def rerank(self,query_text,user,retrieved_docs,retrieved_scores,session=None):
        '''
        Computes the reranking scores based on various arguments. 

        Args:
            query_text: a string containing the searched text.
            user: a string containing the user id. Can be new.
            retrieved_docs: a np.array of strings, representing the doc ids.
            retrieved_scores: a np.array of floats, containing the scores given by ElasticSearch. Index is the same as the docs. May be unsorted.
            session: a string containing the session id. Must support None, i.e. when the sesion is not provided.
        Returns:
            reranked_scores: a np.array of the same shape and type as retrieved_scores, with updated values.
        '''
        if len(retrieved_docs) == 0:
            return None
        queryID = self.getQueryID(query_text)
        self.compute_metrics(user, retrieved_docs)
        return self.mix_scores(
            (retrieved_scores, 1),
            # (self.UD_shortest_distance, 1),
            # (self.UD_weighted_shortest_distance, 1),
            (self.UD_common_neighbors, 1),
            (self.UD_adamic_adar, 1),
            # (self.UD_page_rank, 1),
            # (self.UD_prop_flow, 1)
        )

    def compute_metrics(self, user, retrieved_docs, progress=False):
        time_start = time.time()
        # if progress:
        #     print("Computing user-document metrics\nshortest distance...", end="")
        # self.UD_shortest_distance = np.reciprocal(self.user_document.shortest_distance(user, retrieved_docs))
        # if progress:
        #     print("done\nweighted shortest distance...", end="")
        # self.UD_weighted_shortest_distance = np.reciprocal(self.user_document.weighted_shortest_distance(user, retrieved_docs))
        if progress:
            print("done\ncommon neighbors...", end="")
        self.UD_common_neighbors = self.user_document.common_neighbors_metric(user, retrieved_docs)
        if progress:
            print("done\nadamic adar...", end="")
        self.UD_adamic_adar = self.user_document.adamic_adar_metric(user, retrieved_docs)
        # if progress:
        #     print("done\npage rank...", end="")
        # self.UD_page_rank = self.user_document.rooted_page_rank(user, retrieved_docs)
        # if progress:
        #     print("done\nprop flow...", end="")
        # self.UD_prop_flow = self.user_document.prop_flow(user, retrieved_docs)
        if progress:
            print(f"done\nComputations took {time.time()-time_start}s")

    def evaluation_metrics_scores(self,query_text,user,retrieved_docs,retrieved_scores,session=None):
        '''
        Computes reranking scores based on all metrics chosen for evaluation.

        Args:
            query_text: a string containing the searched text.
            user: a string containing the user id. Can be new.
            retrieved_docs: a np.array of strings, representing the doc ids.
            retrieved_scores: a np.array of floats, containing the scores given by ElasticSearch. Index is the same as the docs. May be unsorted.
            session: a string containing the session id. Must support None, i.e. when the sesion is not provided.
        Returns:
            reranked_scores: a np.array of the shape n_docs x metrics, containing the score for each doc.
            metrics: a np.array containing the unique name of the metrics. Must have the same order as the rows of reranked_scores. 
        '''
        if len(retrieved_docs) == 0:
            return None
        queryID = self.getQueryID(query_text)
        self.compute_metrics(user, retrieved_docs)
        return (
            np.transpose(np.matrix([
                # self.UD_shortest_distance,
                # self.UD_weighted_shortest_distance,
                self.UD_common_neighbors,
                self.UD_adamic_adar,
                # self.UD_page_rank,
                # self.UD_prop_flow
            ])),
            np.array([
                # "user_document-shortest_distance",
                # "user_document-weighted_shortest_distance",
                "user_document-common_neighbors",
                "user_document-adamic_adar",
                # "user_document-page_rank",
                # "user_document-prop_flow",
            ])
        )

    def is_new_user(self,userID): #TODO
        #Must return true if the userId is new, false if it is known
        return not self.user_document.nx_graph.has_node(userID)
    
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


    def PClick(self, queryID, retrieved_docs, retrieved_scores, userID):
        """Hello world metric, defined in https://dl.acm.org/doi/10.1145/1242572.1242651"""
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
                        matches[row[5]] =  (matches.get(row[5], 0) + 1)
        else:
            raise NotImplementedError("Unknown dataset")
        for resultID ,score in zip(retrieved_docs, retrieved_scores):
            newScore = score * (1 - self.ranking_ratio) + (matches.get(resultID, 0)/(potentialMatches+beta)) * self.ranking_ratio
            res.append(newScore)
        return np.array(res)
    
    def mix_scores(self, *arg):
        '''
        Mix reranking scores.

        Args:
            a list of (score, weight) where weight is an float and score is a np.array of shape n_docs x 1
        Returns:
            reranked_scores: a np.array of the shape n_docs x 1, containing the mixed score for each doc.
        '''
        if len(arg) == 0:
            raise ValueError("No metric given.")
        shape = arg[0][0].shape
        res = np.zeros(shape=shape)
        for score, weight in arg:
            assert score.shape == shape
            norm = np.linalg.norm(score)
            normalized_score = score / norm if norm != 0 else np.zeros(shape=score.shape)
            res += weight * normalized_score
        return res
    
    def graph_features(self, query, user, documents, session=None):
        tmp = self.ranking_ratio
        self.ranking_ratio = 1
        k = len(documents)
        res = np.matrix([
            # nodes degrees
            [self.user_document.degree(user)] * k,
            [self.user_document.degree(document) for document in documents],
            [self.session_document.degree(session)] * k,
            [self.session_document.degree(document) for document in documents],
            [self.query_document.degree(query)] * k,
            [self.query_document.degree(document) for document in documents],
            [self.query_user.degree(query)] * k,
            [self.query_user.degree(user)] * k,
            [self.query_session.degree(query)] * k,
            [self.query_session.degree(session)] * k,
            # shortest distance
            self.UD_shortest_distance,
            self.SD_shortest_distance,
            self.QD_shortest_distance,
            [self.QU_shortest_distance] * k,
            [self.QS_shortest_distance] * k,
            # weighted shortest distance (probably don't need both shortest distance metrics)
            self.UD_weighted_shortest_distance,
            self.SD_weighted_shortest_distance,
            self.QD_weighted_shortest_distance,
            [self.QU_weighted_shortest_distance] * k,
            [self.QS_weighted_shortest_distance] * k,
           # Common neighbors
            self.UD_common_neighbors,
            self.SD_common_neighbors,
            self.QD_common_neighbors,
            [self.QU_common_neighbors] * k,
            [self.QS_common_neighbors] * k,
            # Adamic Adar
            self.UD_adamic_adar,
            self.SD_adamic_adar,
            self.QD_adamic_adar,
            [self.QU_adamic_adar] * k,
            [self.QS_adamic_adar] * k,
            # page rank
            self.UD_page_rank,
            self.SD_page_rank,
            self.QD_page_rank,
            [self.QU_page_rank] * k,
            [self.QS_page_rank] * k,
            # prop flow
            self.UD_prop_flow,
            self.SD_prop_flow,
            self.QD_prop_flow,
            [self.QU_prop_flow] * k,
            [self.QS_prop_flow] * k,
        ])
        self.ranking_ratio = tmp
        return res
    