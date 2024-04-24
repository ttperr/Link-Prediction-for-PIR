import argparse
import sys
from indexer import index_in_ElasticSearch
import Reranker
import connector

from UI import Ui_Widget
from PySide6.QtWidgets import QApplication, QWidget

class PIR(object):
    def __init__(self,dataset):
        self.dataset=dataset
        if(dataset=="AOL"):
            self.index="aol4ps"
        else:
            raise NotImplementedError("Unknown dataset")

        self.client = connector.establish_connection()

        self.reranker=Reranker.Reranker(dataset) #In future we can add flags initializing different rerankers (graph, contents,...)

    def is_new_user(self,userId):
        return self.reranker.is_new_user(userId)
    
    def query(self,search_text,user):
        results=self.query_es(search_text)
        clean_results=self.clean_query(results)
        print("ElasticSearch results")
        print(clean_results)
        reranked_results=self.reranker.rerank(search_text,clean_results,user)
        print("Reranked results")
        print(reranked_results)
        return reranked_results #TODO
    
    def query_es(self,search_text):
        if self.dataset=="AOL":
            return self.client.search(index=self.index,query={"match":{"title":search_text}})
        return None
    def clean_query(self,query_result):
        return([(a["_id"],a["_score"]) for a in query_result["hits"]["hits"]])

class Widget(QWidget):
    def __init__(self,PIR, parent=None):
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self,PIR)

def main():
    parser = argparse.ArgumentParser(description="Personalized Information Retrieval project")
    parser.add_argument("-l",default=False,action='store_true',help="Load data to ElasticSearch")
    parser.add_argument("-d",type=str,default="AOL",help="Specify the dataset type. All parameters associated to dataset are hardcoded")
    arguments = parser.parse_args()
    if(arguments.l):
        index_in_ElasticSearch(arguments.d)

    ## THIS IS JUST TO HAVE SOMETHING TO TEST ON FOR NOW, UNTIL THE GUI IS COMPLETE
    pir=PIR(arguments.d)
    app = QApplication(sys.argv)
    widget = Widget(pir)
    widget.show()
    sys.exit(app.exec())




if __name__ =="__main__":
    main()


