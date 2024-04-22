#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE


vocab = open("/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/brown_vocab_100.txt")

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word = line.strip().lower()  
    word_index_dict[word] = i

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict), dtype=int) # TODO: minitialize counts to a zero vector

#TODO: iterate through file and update counts
for line in f:
    words = line.lower().split()
    for word in words:
        if word in word_index_dict:
            counts[word_index_dict[word]] +=1

f.close()

#TODO: normalize and writeout counts. 
normalized_counts = counts / np.sum(counts)

print(counts)
print(normalized_counts)





