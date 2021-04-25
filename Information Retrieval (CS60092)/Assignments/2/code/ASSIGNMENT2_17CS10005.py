from bs4 import BeautifulSoup
import os, re, sys
import errno
from os import listdir
import json
from os.path import isfile, join
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import string
import nltk
import os
import numpy as np
import pickle5 as pickle
from collections import OrderedDict

# nltk.download('all')

def DocumentSearchList(List,positional_index):
    documentSearchList = []
    for t in List:
        for e in positional_index[t]:
            if e[0] not in documentSearchList:
                documentSearchList.append(e[0])
    return documentSearchList

def DotProduct(v1,v2):
    if(np.sum(v1) == 0 or np.sum(v2) == 0):
        return 0
    unit_vector_1, unit_vector_2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
    return np.dot(unit_vector_1, unit_vector_2)

def Score(documentSearchList,queryIDF, documentVectors):
    score = {}
    for Id in documentSearchList:
        docMap = {}
        for e in documentVectors[Id]:
            docMap[e[0]] = e[1]

        qVec, dVec = [],[]

        l= list(queryIDF.keys())
        for term in docMap.keys():
            l.append(term)
        
        uniqueTerms = list(set(l))

        for term in uniqueTerms:
            if term in queryIDF:
                qVec.append(queryIDF[term])
            else:
                qVec.append(0)

        for term in uniqueTerms:
            if term in docMap:
                dVec.append(docMap[term])
            else:
                dVec.append(0)
        score[Id] = DotProduct(qVec,dVec)
    return score

def QualityScores(): 
    path='../Dataset'
    files = [f for f in listdir(path) if f != "." and f != ".."]

    return files, pickle.load(open('../StaticQualityScore.pkl', "rb"))

def BuildDictionary(files):
    d_tf = {}
    file_idf = {}
    filecount = 0
    path='../Dataset/'
    for filename in files:

        file_id = int(filename.split('.',1)[0])
        soup = BeautifulSoup(open(path+filename, encoding='utf8'), "html.parser")

        tf = {}
        simpleText = ""
        for s in soup.find_all('p'):
            simpleText =simpleText +" "+ s.text.strip()
        simpleText=simpleText[1:]
        
        # tokens = word_tokenize(simpleText)
        tokens=[]
        for w in word_tokenize(simpleText):
            tokens.append(w.lower())

        table = str.maketrans('', '', string.punctuation)
        stripped = []
        for w in tokens:
            stripped.append(w.translate(table))
        
        words=[]
        for word in stripped:
            if word.isalpha():
                words.append(word)

        stop_words = set(stopwords.words('english'))
        twords = []
        for w in words:
            if not w in stop_words and w != "":
                twords.append(w)
        words=twords

        lemmatized_words = LemmatizedWords(nltk.pos_tag(words))

        wordCount = 0
        
        for w in lemmatized_words:
            if w in tf.keys():
                tf[w] += 1
            else:
                tf[w] = 1
            wordCount += 1

        for term in tf.keys():
            tf[term] = 1 + np.log(tf[term])
            if term not in d_tf.keys():
                d_tf[term] = []
            d_tf[term].append((file_id,tf[term]))
        
        unique_words = np.unique(np.array(lemmatized_words))
        for w in unique_words:
            if w not in file_idf.keys():
                file_idf[w] = 0 
            file_idf[w] += 1
        filecount += 1        
        print("Processed file_ID: {} \t Total files processed: {}".format(file_id,filecount))

    return d_tf,file_idf 

def InvertedPositionalIndex(d_tf, idf):
    inverted_index={}
    for term in d_tf.keys():
        temp = np.log(1000/idf[term])
        inverted_index[(term,temp)] = [d_tf[term]]
        idf[term] = temp
    
    return OrderedDict(sorted(inverted_index.items()))


def ChampionList_Local(d_tf):
    championLocal = {}
    for term in d_tf:
        championLocal[term] = sorted(d_tf[term], key = lambda x: x[1], reverse = True)[:50] #top 50 

    return championLocal

def ChampionList_Global(d_tf, qualityScores, idf):
    newDTF = {}
    championGlobal = {}
    for term in d_tf:
        l = []
        for elem in d_tf[term]:
            l.append((elem[0],qualityScores[elem[0]] + elem[1] * idf[term]))
        
        newDTF[term]=l

    for term in newDTF:
        championGlobal[term] = sorted(newDTF[term], key = lambda x: x[1], reverse = True)[:50] #top 50
    
    return championGlobal


def BuildocVector(inverted_index):
    t_tfIDF = {} #pair[0] -> (docId,tf_idf score)
    t_idf = {} #pair[0] -> (idf score)

    print("################ Building Document Vector ###################")
    documentVectors = {} #docId -> (pair[0],tf_idf)
    for pair in inverted_index.keys():
        t_tfIDF[pair[0]] = []
        
        for elem in inverted_index[pair][0]:
            tf_idf = elem[1] * pair[1]
            docId = elem[0]
            t_tfIDF[pair[0]].append((docId,tf_idf))
            if docId not in documentVectors.keys():
                documentVectors[docId] = []
            documentVectors[docId].append((pair[0],tf_idf)) 

        t_idf[pair[0]] = pair[1]

    print("################ Document Vector built ################")
    return t_tfIDF, t_idf, documentVectors

def BuildLeaderFollowerDict(documentVectors, leaders):
    followersDict = {}
    followersList = [] 

    followersList = [docId for docId in documentVectors.keys() if docId not in leaders]

    print("#################### Building Leader:Followers Dictionary #####################")
    count = 0
    for docId in followersList:
        print("Processing Follower: ",count)
        count += 1

        followerVector = {}
        for elem in documentVectors[docId]:
            followerVector[elem[0]] = elem[1] 

        leaderScores = Score(leaders,followerVector,documentVectors)
        nearestLeader = max(leaderScores, key=leaderScores.get)

        if nearestLeader not in followersDict.keys():
            followersDict[nearestLeader]=[]
        followersDict[nearestLeader].append(docId)

    print("#################### Leader:Followers Dictionary built ####################")

    return followersDict, leaderScores, nearestLeader

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

def PrintTop10(l, str):
    print(str)
    for elem in l:
        if elem[1] > 0:
            print(elem)
    print()

def ProcessQueries(queries, championLocal, championGlobal, t_tfIDF, t_idf, documentVectors, followersDict, leaderScores, nearestLeader, leaders):
    queryFile = open(queries, 'r')
    fp = open('RESULTS2_17CS10005.txt','w')
    for query in queryFile.readlines():
        if(query == "\n"):
            continue

        # tokens = word_tokenize(query)
        tokens = []
        for w in word_tokenize(query):
            tokens.append(w.lower())

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        stripped = []
        for w in tokens:
            stripped.append(w.translate(table))

        # remove remaining tokens that are not alphabetic
        words = []
        for word in stripped:
            if word.isalpha():
                words.append(word)

        # filter out stop words
        stop_words = set(stopwords.words('english'))
        twords = []
        for w in words:
            if not w in stop_words and w != "":
                twords.append(w)
        
        words=twords

        tags = nltk.pos_tag(words)
        lemmatized_words = LemmatizedWords(tags)

        #V(Q)(t) = idft
        queryIDF = {}
        for term in lemmatized_words:
            queryIDF[term] = t_idf[term]
        
        print("For query:",query,end = "")
        
        #tf_idf score (Q,d)
        documentSearchList = DocumentSearchList(lemmatized_words,t_tfIDF)
        q_score = Score(documentSearchList,queryIDF,documentVectors)
        q_top10 = sorted(q_score.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)[:10]
        PrintTop10(q_top10,"Top 10 tf_idf_scores are:")

        documentSearchList = DocumentSearchList(lemmatized_words,championLocal)
        championLocal_score = Score(documentSearchList,queryIDF,documentVectors)
        championLocal_top10 = sorted(championLocal_score.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)[:10]
        PrintTop10(championLocal_top10,"Top 10 Local_Champion_score are:")

        documentSearchList = DocumentSearchList(lemmatized_words,championGlobal)
        championGlobal_score = Score(documentSearchList,queryIDF,documentVectors)
        championGlobal_top10 = sorted(championGlobal_score.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)[:10]
        PrintTop10(championGlobal_top10, "Top 10 Global_Champion_score are:")
 

        #Cluster pruning scheme
        #Get leader nearest to Query
        leaderQueryScore = Score(leaders,queryIDF,documentVectors)
        nearestLeader = max(leaderScores, key=leaderScores.get)

        #Get cluster pruning score
        documentSearchList = followersDict[nearestLeader]
        documentSearchList.append(nearestLeader)
        clusterPruningScore = Score(documentSearchList,queryIDF,documentVectors)
        clusterPruning_top10 = sorted(clusterPruningScore.items(), key = lambda kv:(kv[1], kv[0]), reverse = True)[:10]
        PrintTop10(clusterPruning_top10, "Top 10 Cluster_pruning_score are:")

        print("\n")
        fp.write(query)
        fp.write(', '.join('<{},{}>'.format(x[0],x[1]) for x in q_top10))
        fp.write('\n')
        fp.write(', '.join('<{},{}>'.format(x[0],x[1]) for x in championLocal_top10))
        fp.write('\n')
        fp.write(', '.join('<{},{}>'.format(x[0],x[1]) for x in championGlobal_top10))
        fp.write('\n')
        fp.write(', '.join('<{},{}>'.format(x[0],x[1]) for x in clusterPruning_top10))
        fp.write('\n\n')
    fp.close()

def main():
    files, qualityScores = QualityScores()
    d_tf, idf = BuildDictionary(files)
    inverted_index = InvertedPositionalIndex(d_tf, idf)

    championLocal=ChampionList_Local(d_tf)
    championGlobal=ChampionList_Global(d_tf, qualityScores,idf)

    #---------------ANSWERING QUERIES--------------#
    t_tfIDF, t_idf, documentVectors = BuildocVector(inverted_index)

    leaders = pickle.load(open('../Leaders.pkl',"rb"))
    followersDict, leaderScores, nearestLeader = BuildLeaderFollowerDict(documentVectors, leaders)


    queries = sys.argv[1]
    ProcessQueries(queries, championLocal, championGlobal, t_tfIDF, t_idf, documentVectors, followersDict, leaderScores, nearestLeader, leaders)

if __name__=='__main__':
    main()