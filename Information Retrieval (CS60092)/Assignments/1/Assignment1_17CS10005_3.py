import os
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pickle

#nltk.download('stopwords')

def generatelist(postingList, file , boolean):
    f = open(os.path.join("ECTText", file))
    sent = f.read().replace("\n", " ").lower()

    tokeniser = RegexpTokenizer(r'\w+')
    tokens = tokeniser.tokenize(sent)

    sw = stopwords.words('english')
    filtered_words = [w for w in tokens if not w in sw]
    
    docID = str(file).split("-")[0]
    ind=0
    flag=-1

    for word in filtered_words:
        boolean=True
        flag=1
        if word not in postingList:
            postingList[word]=[]
            flag=-1
        postingList[word].append((docID, ind))
        ind = ind+1
    
    print(docID+" completed" )
  

def main():
    data_dir = "ECTText"
    pickelfile = "newPositionalIndex.pkl"

    postingList = {}
    boolean=True
    for file in os.listdir(data_dir):
        generatelist(postingList, file, boolean)
   
    f = open(pickelfile, "wb")
    pickle.dump(postingList, f)
    f.close() 

if __name__=='__main__':
    main()