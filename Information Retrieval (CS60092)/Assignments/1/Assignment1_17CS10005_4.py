import pickle
import os
import sys

resultFile= open("RESULTS1_17CS10005.txt","w+")
file = open("newPositionalIndex.pkl", "rb")
invertedDictfile = pickle.load(file)

queries = []
inverted = []


with open(sys.argv[1]) as f:
    for line in f:
        temp = line.split()
        queries.append(temp[0])

for w in invertedDictfile.keys():
    for i in range(len(w)+1):
        wr=w[i:]+"$"
        wr=wr+w[0:i]
        inverted.append([wr, w])    

for query in queries:
    # query = q[:-1] 
    # print(query,query[:-1])
    lst=query.split("*")
    perm=query+"$"

    if len(lst) is 2:
        perm=lst[1]
        perm=perm+"$"
        perm=perm+lst[0]
        perm=perm+"*"

    matches = []
    for pt in inverted:
        w=pt[0]
        endstar=0
        if perm[len(perm)-1]=="*":
            endstar=1

        if endstar==1:
            if w.startswith(perm[:-1]):
                matches.append(pt[1])
        else:
             if w.startswith(perm):
                matches.append(pt[1])
    
    matches=set(matches)
    resultFile.write(query)
    resultFile.write("\n")

    res=""
    for match in matches:
        for element in invertedDictfile[match]:
            res+="("+ element[0] + ", "+ str(element[1]) +"), "
    resultFile.write(res[:-2]+"\n")

f.close()
resultFile.close()
