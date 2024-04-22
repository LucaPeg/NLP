#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = open("/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/brown_vocab_100.txt")
output_filename = "/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/smooth_probs.txt"

#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word = line.strip().lower()  
    word_index_dict[word] = i
    

f = open("/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/brown_100.txt")


counts = np.zeros((len(word_index_dict), len(word_index_dict)), dtype=float) #TODO: initialize numpy 0s array

#TODO: iterate through file and update counts
previous_word = '<s>'
for line in f:
    words = line.lower().split()
    for word in words:
        if previous_word in word_index_dict and word in word_index_dict:
            counts[word_index_dict[previous_word], word_index_dict[word]] +=1
        previous_word = word
f.close()

counts += 0.1

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
with open(output_filename, 'w') as wf:
    wf.write(f"P(the | all) = {probs[word_index_dict['all'], word_index_dict['the']]}\n")
    wf.write(f"P(jury | the) = {probs[word_index_dict['the'], word_index_dict['jury']]}\n")
    wf.write(f"P(campaign | the) = {probs[word_index_dict['the'], word_index_dict['campaign']]}\n")
    wf.write(f"P(calls | anonymous) = {probs[word_index_dict['anonymous'], word_index_dict['calls']]}")