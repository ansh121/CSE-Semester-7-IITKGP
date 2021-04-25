import os
from os import listdir
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import RegexpTokenizer     
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import f1_score
import sys
import numpy as np
import math


path = sys.argv[1]
c1path = path + "/class1/"
c2path = path + "/class2/"

path += "/"
testc1path = c1path + "test/"
testc2path = c2path + "test/"


print("start ...\n")
stop_words = set(stopwords.words('english'))

trainc1path = c1path + "train/"
trainc2path = c2path + "train/"

outpath = sys.argv[2]
outpath += ".txt"


def LemmatizedWords(tags):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for t in tags:
        posTag = ''
        if t[1].startswith('J'):
            posTag = wordnet.ADJ
        elif t[1].startswith('V'):
            posTag = wordnet.VERB
        elif t[1].startswith('R'):
            posTag = wordnet.ADV
        elif t[1].startswith('N'):
            posTag = wordnet.NOUN
        l = ''
        if posTag == '':
            l = lemmatizer.lemmatize(t[0])
        else:
            l = lemmatizer.lemmatize(t[0],posTag)
        lemmatized_words.append(l)
    return lemmatized_words

def normalize(text):
    word_pos = nltk.pos_tag(nltk.word_tokenize(text))
    lemm_words=LemmatizedWords(word_pos)

    return [x.lower() for x in lemm_words]

def clearfile(path):
    val=1
    with open(path, 'rb') as f:
        contents = str(f.read())
        val=val+1
    contents = contents.replace('\\n',' ')
    val=3
    contents = contents.replace('\\', '')
    val=4
    contents = contents.replace('\'','')
    return val, contents

def QualityScores(path): 
    files = [f for f in listdir(path)]

    return files

def join_words(tokens):
    t=[w for w in tokens if not w in stop_words]
    l=" ".join(t)

    return l

def update_newd(lemList, newd):
    for term in lemList:
        if term in newd.keys():
            newd[term] += 1
        else:
            newd[term] = 1

def documentVector(path):
    tf={} 
    dc=[]
    for f in QualityScores(path):
        #print(f)
        filepath = os.path.join(path,f)
        tokenizer = RegexpTokenizer(r'\w+')
        _, contents=clearfile(filepath)
 
        newd={}
        word_tokens=tokenizer.tokenize(contents)

        filtered_st = join_words(word_tokens)

        lemList =  normalize(filtered_st)

        update_newd(lemList, newd)

        unique_words = np.unique(np.array(lemList))

        update_newd(unique_words, tf)
        dc.append(newd)
    
    return tf,dc

def get_terms(term,tfC1,tfC2):
    N10 = 0    
    N11 = tfC1[term]
    N01 = N1-N11
    if term in tfC2.keys():
        N10 = tfC2[term]
    N00 = N2 - N10
    N1d = (N10+N11)
    N0d = (N01 + N00)
    Nd1 = (N11 + N01)
    Nd0 = (N00 + N10)

    return N10,N11,N01,N00,N1d,N0d,Nd1,Nd0

def update_term_value(term,N10,N11,N01,N00,N1d,N0d,Nd1,Nd0,N,miC1):
    miC1[term] = (N01/N)*math.log2((N*N01)/(N0d*Nd1)) + (N11/N)*math.log2((N*N11)/(N1d*Nd1)) 
    val=1
    if N10>0:
        miC1[term] += (N10/N)*math.log2((N*N10)/(N1d*Nd0)) 
    else:
        val=val+1
    try:
        miC1[term] += (N00/N)*math.log2((N*N00)/(N0d*Nd0))
    except:
        hmm=True

def getmiC1(tfC1,tfC2,N):
    miC1={}
    for term in tfC1.keys():
        N10,N11,N01,N00,N1d,N0d,Nd1,Nd0=get_terms(term, tfC1, tfC2)
        update_term_value(term,N10,N11,N01,N00,N1d,N0d,Nd1,Nd0,N,miC1)

    return miC1

def getmiC2(tfC2,tfC1,N):
    miC2={}
    for term in tfC2.keys():
        N10,N11,N01,N00,N1d,N0d,Nd1,Nd0=get_terms(term,tfC2, tfC1)
        update_term_value(term,N10,N11,N01,N00,N1d,N0d,Nd1,Nd0,N,miC2)

    return miC2

def top10000(miC1):
    l=[]
    for k, v in sorted(miC1.items(), key=lambda item: item[1], reverse=True):
        l.append(k)
    return l[:10000]

# tfC1= {}#term -> freq(#docs the term appears)
# tfC2= {}#term -> freq
# dvectrainC1=[] #list of document vectors
# dvectrainC2=[] #list of doc vecs
tfC1,dvectrainC1 = documentVector(trainc1path)
tfC2,dvectrainC2 = documentVector(trainc2path)

f1_m = []
f1_b = []

# docVecTestC1=[] #list of document vectors
# docVecTestC2=[] #list of doc vecs

counts=[1, 10, 100, 1000, 10000]

N1=len(os.listdir(trainc1path))
N2=len(os.listdir(trainc2path))
N=N1+N2

_, docVecTestC1 = documentVector(testc1path)
_, docVecTestC2 = documentVector(testc2path)



#Mutual information of terms of class c1
miC1=getmiC1(tfC1,tfC2,N) #term->MI.
miC2=getmiC2(tfC2,tfC1,N) #term->MI.


top10000C1 = top10000(miC1)

def generate_data(doc1, doc2, features):
    t=[]
    for dic in doc1:
        l=[]
        for f in features:
            l.append(dic.get(f, 0))
        t.append(l)

    for dic in doc2:
        l=[]
        for f in features:
            l.append(dic.get(f, 0))
        t.append(l)

    return t



for n in counts:
    features = top10000C1[:n]
    #making dataset
    
    trainsetx=generate_data(dvectrainC1, dvectrainC2, features)
    testsetx=generate_data(docVecTestC1, docVecTestC2, features)

    ytrain = [1] * N1 + [2] * N2
    ytest = [1] * len(docVecTestC1) + [2] * len(docVecTestC2)
    model = MultinomialNB()
    model.fit(trainsetx, ytrain)
    y_pred=model.predict(testsetx)
    f1s=f1_score(ytest, y_pred, average='micro')
    f1_m.append(f1s)

    model2 = BernoulliNB()
    model2.fit(trainsetx, ytrain)
    y_pred=model2.predict(testsetx)
    f1s=f1_score(ytest, y_pred, average='micro')
    f1_b.append(f1s)



with open(outpath, 'w') as file:
    file.write("OUTPUT FILE 1\n")
    file.write("#Features\t1\t10\t100\t1000\t10000\n")
    file.write(f"Multinomial_NB\t{round(f1_m[0],3)}\t{round(f1_m[1],3)}\t{round(f1_m[2],3)}\t{round(f1_m[3],3)}\t{round(f1_m[4],3)}\n")
    file.write(f"Bernoulli_NB\t{round(f1_b[0],3)}\t{round(f1_b[1],3)}\t{round(f1_b[2],3)}\t{round(f1_b[3],3)}\t{round(f1_b[4],3)}")

print("finish\n")