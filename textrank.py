from sklearn.neighbors import DistanceMetric
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity  

import codecs
import random


def read_data(filename):
	f = codecs.open(filename, encoding="utf-8", mode="r")
	text = [t.strip() for t in f.readlines()]
	return text

# to implement later
def prepare_data(text):
	return text

def cal_similarity(text, sim_method="euclidean"):
	vectorizer = TfidfVectorizer(ngram_range=(1, 3))#, min_df=.15, max_df=.85)
	tfidf = vectorizer.fit_transform(text)

	return cosine_similarity(tfidf)
	# dist = DistanceMetric.get_metric(sim_method)
	# return dist.pairwise(tfidf)

def run_textrank(sim_mat, d=.85):
    
    
    sums = [sum(s) - 1 for s in sim_mat]
    
    # list of text and random weight for it
    updated_weights = [random.uniform(1, 100) for t in sim_mat]
    pre_weights = [0] * len(sim_mat)
    
    # iterate to 1000 if no convergence happened
    itr = 0
    is_converged = False
    while iter < range(1000) and not is_converged:

        # is_converged = True
        for i in range(len(updated_weights)):
            pre_weights[i] = updated_weights[i]
            temp = 0
            for j in range(len(updated_weights)):
                if i != j:
#                     temp += (sim_mat[i][j] / (sum(sim_mat[j]) - 1)) * updated_weights[j]
                    temp += (sim_mat[i][j] / sums[j]) * updated_weights[j]
            updated_weights[i] = round((1-d) + d * temp, 4)#(1 - d) * d * temp

        is_converged = True
        for i in range(len(updated_weights)):
            if abs(pre_weights[i] - updated_weights[i]) > .0001:
                is_converged = False
                break

        itr += 1
    return sorted([(updated_weights[i], i) for i in range(len(updated_weights))], reverse=True), itr


def workflow(filename="input_test.txt"):
	text = read_data(filename)
	cleaned_text = prepare_data(text)
	sim_mat = cal_similarity(text)
	rankedlist, itr_no =run_textrank(sim_mat)
	print rankedlist
	return rankedlist, itr_no

workflow()