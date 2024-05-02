import csv
from tqdm import tqdm
import datetime

class LogManager(object):
    def __init__(self,dataset):
        if dataset!="AOL":
            raise NotImplementedError("Unknown dataset")
        self.dataset=dataset
        self.user_session={}
        self.new_user=1
        self.quey_ids={}
        self.last_user=-1
        self.load_users()

    def load_users(self):
        if self.dataset!="AOL":
            raise NotImplementedError("Unknown dataset")
        with open("datasets/AOL4PS/data.csv") as f:
                reader = csv.reader(f, delimiter='\t')
                firstRow = True
                pbar = tqdm(reader, desc='Parsing queries for log manager', unit='rows')
                for row in pbar:
                    if firstRow:
                        firstRow = False
                        continue
                    self.user_session[int(row[0])]=int(row[3])
                    self.new_user=max(self.new_user,int(row[0])+1)
    def get_new_user(self):
        return str(self.new_user)
    def is_new_user(self,userId):
        return not userId in self.user_session
    
    def getQueryID(self, query_text):
        """parse query.csv and return matching query id and if the id is new"""
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
                        return row[1],False
                    idNumber = int(row[1][2:])
                    if idNumber > maxId:
                        maxId = idNumber
        else:
            raise NotImplementedError("Unknown dataset")
        return f"q-{maxId+1}",True
    
    def getQueryText(self, query_id):
        line=int(query_id[2:])+1
        with open('datasets/AOL4PS/query.csv') as f:
            lines=f.readlines()
            return lines[line].split("\t")[0]
    def register_log(self,doc_ids, user_id, doc_clicked_index, query_text):
        try:
            int(user_id)
        except:
            print("To save logs the User field must be an integer number. If you are a new user, please use",self.get_new_user()," or any integer displayed in green.")
            return False,False,False
        if self.dataset!="AOL":
            raise NotImplementedError("Unknown dataset")
        
        if self.is_new_user(int(user_id)):
            print("New user detected")
            self.user_session[int(user_id)]=-1
            if(int(user_id)==self.new_user):
                self.new_user+=1
        if user_id!=self.last_user:
            self.user_session[int(user_id)]+=1
            self.last_user=user_id
        sessionId=self.user_session[int(user_id)]
        queryId,isQueryNew=self.getQueryID(query_text)
        if isQueryNew:
            print("New query detected")
            with open("datasets/AOL4PS/query.csv","a") as f:
                f.write("\n"+query_text+"\t"+queryId)
                pass
        top_ten_docs=doc_ids[:10]
        timestamp=datetime.datetime.now().replace(microsecond=0)
        if doc_clicked_index>9:
            top_ten_docs[9]=doc_ids[doc_clicked_index]
            doc_clicked_index=9
        docs="\t".join(top_ten_docs)
        line=f'\n{user_id}\t{queryId}\t{timestamp}\t{sessionId}\t{top_ten_docs[doc_clicked_index]}\t\"{docs}\"\t{doc_clicked_index}'
        print(line)
        with open("datasets/AOL4PS/data.csv","a") as f:
                f.write(line)
        return queryId,str(sessionId),True