#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""
import numpy as np
import os
os.chdir('A2')
from generate import GENERATE

vocab = open('brown_vocab_100.txt')

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

# Normalize counts to get probabilities
total_count = np.sum(counts)
probabilities = counts / total_count if total_count > 0 else counts

print(counts)
print(normalized_counts)

# LUCA'S UGLY PART ####################################

single_occurrence_count = np.sum(counts == 1)
total_word_types = len(word_index_dict)
prop_single_occurrence = single_occurrence_count / total_word_types
print(prop_single_occurrence) # 65% of words appear only once

# what if we take the whole corpus?

from nltk.corpus import brown
from collections import Counter

words = brown.words()
word_counts = Counter(words)
total_word_types = len(word_counts)
single_occ_tot = sum(1 for count in word_counts.values() if count == 1)
proportion_single_occurrence = single_occ_tot / total_word_types
print(f"Proportion single occurrence: {proportion_single_occurrence:.2f}")

# QUESTION 6 ##########################################

with open('toy_corpus.txt', 'r') as corpus, open('unigram_eval.txt', 'w') as output:
    for line in corpus:
        words = line.strip().lower().split()
        sent_prob = 1 # initialize probability before looping the words
        sent_len = len(words) 

        for word in words:
            if word in word_index_dict:
                word_prob = probabilities[word_index_dict[word]]
            else:
                word_prob = 0 
            sent_prob *= word_prob

        output.write(f"{sent_prob}\n") # Computing Pr just to check results
    
        # Calculate perplexity
        perplexity = 1 / (pow(sent_prob, 1.0 / sent_len)) if sent_prob > 0 else float('inf')
        
        # Write perplexity to file
        output.write(f"{perplexity}\n")

        # Optional: print each sentence's perplexity for verification
        print(f"Perplexity of the sentence: {perplexity}")

# QUESTION 7 ###############################################

with open('unigram_sentences', 'w') as output:
    for i in range(1,11):
        gen_text = GENERATE(word_index_dict, probabilities, "unigram", 15, "County")
        output.write(f"{gen_text}\n")