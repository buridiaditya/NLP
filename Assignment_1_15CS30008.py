import nltk 
from nltk.corpus import brown
from string import punctuation
dataset = brown.sents()


dataProcessed = [ [word.lower() for word in sentence if word not in punctuation] for sentence in dataset ]
'''
for lineno in range(len(dataset)):
    for wordno in range(len(dataset[lineno])):
        ar = []
        ar = ar.append(dataset[lineno][wordno].lower())
    dataProcessed = dataProcessed.append(ar)
'''

train = dataProcessed[0:40000]
test = dataProcessed[40000:]
print(train,test)
