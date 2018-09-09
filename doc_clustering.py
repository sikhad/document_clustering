# Simple document clustering implementation

import re
import numpy as np
import random
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import cdist

random.seed(42834)

# read file
f = open("test.txt")
full_text = f.readlines()[0]
f.close()

documents = full_text.split()

# check number of documentions
print(len(documents))
print('\n')

# preprocess documents (remove special characters and lowercase everything)
for i in range(len(documents)):
	documents[i] = " ".join(documents[i].split())
	documents[i] = documents[i].replace(r"\[.*\]","")
	documents[i] = re.sub(r'([^\s\w]|_)+', '', documents[i])
	documents[i] = documents[i].lower()

# use tfidf to create word vector
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# try different k values
k = 7
model = KMeans(n_clusters=k, init='k-means++', max_iter=1000, n_init=1)
model.fit(X)
 
# print top terms per cluster (code snippet taken from python documentation online)
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :50]:
        print(' %s' % terms[ind]),
    print
    print('\n')

# find which cluster each document is a part of
Y = vectorizer.transform(documents)
preds = model.predict(Y)
print(preds)
print('\n')

# calculate percent of documents in each cluster
count = Counter(preds)
total = sum(count.values(), 0.0)
for key in count:
    count[key] =  count[key]/total

print(count)

# output documents to file per cluster
preds = preds.tolist()

for num in range(k):
	count = 0
	documents_write = ''
	for i in range(len(preds)):
		if num==preds[i]:
			count += 1
			documents_write += str(documents[i])
	f2 = open("output_"+str(num)+".txt","w")
	f2.write(documents_write)
	f2.close()