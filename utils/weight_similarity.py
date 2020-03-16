import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import spatial
import operator

weights = np.load('../weights.npy')
ind_to_tag = ['PAD', 'B-ORG', 'O', 'B-MISC', 'B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC']
def similarity(w1, w2, combination):
    '''
    Cosine similarity is 1-distance
    :param w1: row in weight matrix of first tag
    :param w2: row in weight matrix of tag 2
    :return: cosine similarity between them
    '''
    ##print("combination = {}".format(combination))
    ##print("combination = {}\n   row 1: {}\n row 2: {}".format(combination, w1, w2))
    # 1 - dist is similarity
    sim = 1- spatial.distance.cosine(w1, w2)
    # print("similarity: {}".format(sim))
    return sim


def print_sims(sorted_sim_dict):
    for comb, sim in sorted_sim_dict.items():
        print("{}\t{}\t{}".format(ind_to_tag[comb[0]], ind_to_tag[comb[1]], sim))


comb = combinations([1,2,3,4,5,6,7,8,9],2)  # omit 0, the padding tag
sim_dict = {}
sims = {}
for c in list(comb):
    s = similarity(weights[c[0]], weights[c[1]], c)
    sim_dict[c] = s
    sims[s] = c

# sort by descending similarity
sorted_d = dict(sorted(sim_dict.items(), key=operator.itemgetter(1), reverse=True))

# Does this have a bug because getting negative numbers?
print_sims(sorted_d)



