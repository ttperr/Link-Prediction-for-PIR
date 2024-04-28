import csv
import numpy as np
from tqdm import tqdm
from graph import Graph

class Reranker(object):

    def __init__(self,dataset,validation=False):
        self.dataset = dataset
        self.validation=validation
        if dataset=="AOL":
            #YOU SHOULD LOAD VARIABLES/USEFUL INFO FROM LOGS OR A FILE STORING THEM (IF YOU USE FILES, STORE THEM IN THE CORRECT FOLDER)
            #THE FOLDER with data and files needed to rerank SHOULD BE in  AOL4PS/ 
            self.user_document = self.generate_user_document_graph()
            
        else:
            raise NotImplementedError("Unknown dataset")

    def rerank(self,query_text,query_results,user):
        if len(query_results) == 0:
            return []
        queryID = self.getQueryID(query_text)
        # return self.PClick(queryID, query_results, user)
        return self.graph_metric_ranking(user, query_results)

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

    def generate_user_document_graph(self):
        res = Graph()
        if self.dataset=="AOL":
            doc='datasets/AOL4PS/data.csv'
            if self.validation:
                doc='datasets/AOL4PS/training_data.csv'
            with open(doc) as f:
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Parsing queries', unit='rows')
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    res.add_link(row[0], row[5])
        else:
            raise NotImplementedError("Unknown dataset")
        return res
        

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
    
    def graph_metric_ranking(self, user, documents):
        ranking_ratio = 0.7
        res=[]
        for document, score in documents:
            metric = self.user_document.common_neighbors(user, document)
            # metric = self.user_document.adamic_adar(user, document)
            # metric = self.user_document.rooted_page_rank(user, document)
            new_score = score * (1 - ranking_ratio) + metric * ranking_ratio
            res.append((document, new_score))
        return res
    
    


    

