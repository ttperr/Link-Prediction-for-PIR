import numpy as np

class Reranker(object):
    def __init__(self,dataset):
        if dataset=="AOL":
            #YOU SHOULD LOAD VARIABLES/USEFUL INFO FROM LOGS OR A FILE STORING THEM (IF YOU USE FILES, STORE THEM IN THE CORRECT FOLDER)
            #THE FOLDER with data and files needed to rerank SHOULD BE in  AOL4PS/ 
            pass
        else:
            raise NotImplementedError("Unknown dataset")

    def rerank(self,query_text,query_results,user):
        #JUST A RANDOM FUNCTION TO SEE DIFFERENT RESULTS(RANDOM)
        res=[]
        for doc,score in query_results:
            res.append((doc,score*np.random.rand()))
        #NO NEED TO SORT THEM AT THE END, I'LL DO THAT IN THE GUI 
        return res

    def is_new_user(self,userID):
        #Must return true if the userId is new, false if it is known

        #JUST RANDOM FOR NOW
        if np.random.rand()>0.5:
            return True
        return False