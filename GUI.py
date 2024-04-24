import argparse
import sys
from indexer import index_in_ElasticSearch
import Reranker
import connecter

class GUI(object):
    def __init__(self,dataset):
        self.dataset=dataset
        if(dataset=="AOL"):
            self.index="aol4ps"
        else:
            raise NotImplementedError("Unknown dataset")

        self.client = connecter.establish_connection()

        self.reranker=Reranker.Reranker(dataset) #In future we can add flags initializing different rerankers (graph, contents,...)

    def query(self,search_text):
        if self.dataset=="AOL":
            return self.client.search(index=self.index,query={"match":{"title":search_text}})
        return None
    def clean_query(self,query_result):
        return([(a["_id"],a["_score"]) for a in query_result["hits"]["hits"]])

    





def main():
    parser = argparse.ArgumentParser(description="Personalized Information Retrieval project")
    parser.add_argument("-l",default=False,action='store_true',help="Load data to ElasticSearch")
    parser.add_argument("-d",type=str,default="AOL",help="Specify the dataset type. All parameters associated to dataset are hardcoded")
    arguments = parser.parse_args()
    if(arguments.l):
        index_in_ElasticSearch(arguments.d)

    ## THIS IS JUST TO HAVE SOMETHING TO TEST ON FOR NOW, UNTIL THE GUI IS COMPLETE
    gui=GUI(arguments.d)
    text = input("QUERY: (write 'q' to quit)\n")
    while(text!="q"):
        results=gui.query(text)
        print("ElasticSearch values")
        print(gui.clean_query(results))
        print("Reranked results")
        print(gui.reranker.rerank(text,gui.clean_query(results),"WRITE HERE THE USER ID YU ARE TESTING"))
        text=input("QUERY: (write 'q' to quit)\n")



if __name__ =="__main__":
    main()