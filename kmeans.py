import nltk
import spacy
import numpy as np
import random
import math

def readData(file):
    data=[]
    with open(file) as f:
        for linea in f:
            sentence = nltk.word_tokenize(linea)       
            sentence = [word.lower() for word in sentence]
            data.append(sentence)
    f.close()
    return data

def mat_x(data):
    values=[]
    for vector in data:
        values.append(vector[:-2])
    return values

def mat_y(data):
    values=[]
    for vector in data:
        if vector[-1] == "spam":
            values.append(1)
        else:
            values.append(0)
    return values

def get_lems(data):
	lems = []
	nlp = spacy.load('en')
	for vector in data:
		lema= []
		for token in vector:
			doc = nlp(token)
			lem = doc[0].lemma_
			lema.append(lem)
		lems.append(lema)
	return lems

def vocabulary(data):
	vocabulary = []
	for vector in data:
		for word in vector:
			if vocabulary.count(word) == 0:
				vocabulary.append(word)
	return sorted(set(vocabulary))


def num_mat_x(data,voc):
    nums = []
    for vector in data:
        vec_freq = [1] 
        for word in voc:
            f = vector.count(word)
            vec_freq.append(f)
        vec_freq = np.array(vec_freq)
        vec_freq = vec_freq / vec_freq.sum()
        nums.append(vec_freq)
    return nums


def clustering(x_values,iterations,k = 2):
	m = len(x_values)
	centroids = []
	mk = initialize_mk(x_values)
	cn = []
	for i in range(m):
		cn.append(0)

	for j in range(iterations):
		#print("initialize_mk")
		for i in range(m):
			cn[i] = get_closest(x_values[i],mk)
			#print("closest",i)

		mk = get_mk_points(cn,x_values)
		#print("mk points")
		centroids.append(cn)
	return centroids

def initialize_mk(x_values,k = 2):
	mk = []
	for i in range(k):
		mi = x_values[random.randint(0,len(x_values))]
		mk.append(mi)
	return mk	

def distance(x1, x2):
	d = (x1 - x2) ** 2
	return math.sqrt(d.sum())

def get_closest(xi, mk):
	index = 0
	for i in range(len(mk)):
		aux = distance(xi, mk[i])
		if i == 0:
			d = aux
		else:
			if d > aux:
				d = aux
				index = i
	return index

def get_mk_points(centroids, x_values, k = 2):
	mks = []
	for i in range(k):
		mks.append([])

	for i in range(len(centroids)):
		mks[centroids[i]].append(x_values[i])

	for i in range(k):
		aux = np.array(mks[i])
		if len(aux) == 0:
			mks[i] = np.array([x_values[random.randint(0,len(x_values))]])
		
	for i in range(k):
		aux = np.array([])
		for j in range(len(mks[i])):
			if j == 0:
				aux = mks[i][j]
			else:
				aux = aux + mks[i][j]

		mks[i] = aux/len(mks[i])

	return mks

def get_cluster_numbers(centroids, tags):
	clusters = []
	for i in range(len(centroids)):
		numbers = {}
		for tag in tags:
			numbers[tag] = 0
		clusters.append(numbers)

	for i in range(len(centroids)):
		for value in centroids[i]:
			clusters[i][tags[value]] = clusters[i][tags[value]] + 1

	return clusters

def error_table(clusters, y_values,tags):
	cl = len(clusters)
	print("Real values:")
	print("Spam",y_values.count(1))
	print("Ham",y_values.count(0))
	for i in range(cl):
		print("Cluster",i,":")
		for tag in tags:
			print(tag, clusters[i][tag])

if __name__ == '__main__':
	data = readData("data.txt")
	x_values = mat_x(data)
	y_values = mat_y(data)
	print("Data")

	x_values = get_lems(x_values)
	print("Lems")

	voc = vocabulary(x_values)	
	print("Voc")

	x_values = num_mat_x(x_values,voc)
	print("Nums")

	centroids = clustering(x_values,2)
	print("Cluster")
	#print(centroids)

	tags = ["ham","spam"]
	clusters = get_cluster_numbers(centroids, tags)

	error_table(clusters,y_values,tags)