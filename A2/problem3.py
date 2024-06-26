#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
import os
os.chdir('A2')
from sklearn.preprocessing import normalize
from generate import GENERATE
import random


vocab = open("brown_vocab_100.txt")
output_filename = "bigram_probs.txt"


#load the indices dictionary
word_index_dict = {}
for i, line in enumerate(vocab):
    #TODO: import part 1 code to build dictionary
    word = line.strip().lower()  
    word_index_dict[word] = i
    

f = open("brown_100.txt")


counts = np.zeros((len(word_index_dict), len(word_index_dict)), dtype=int) #TODO: initialize numpy 0s array

#TODO: iterate through file and update counts
previous_word = '<s>'
for line in f:
    words = line.lower().split()
    for word in words:
        if previous_word in word_index_dict and word in word_index_dict:
            counts[word_index_dict[previous_word], word_index_dict[word]] +=1
        previous_word = word
f.close()

#TODO: normalize counts
probs = normalize(counts, norm='l1', axis=1)

#TODO: writeout bigram probabilities
with open(output_filename, 'w') as wf:
    wf.write(f"P(the | all) = {probs[word_index_dict['all'], word_index_dict['the']]}\n")
    wf.write(f"P(jury | the) = {probs[word_index_dict['the'], word_index_dict['jury']]}\n")
    wf.write(f"P(campaign | the) = {probs[word_index_dict['the'], word_index_dict['campaign']]}\n")
    wf.write(f"P(calls | anonymous) = {probs[word_index_dict['anonymous'], word_index_dict['calls']]}")


# QUESTION 6 #####################################################

# Process the toy corpus to calculate perplexity for each sentence
with open("toy_corpus.txt", 'r') as corpus, open("bigram_eval.txt", 'w') as output:
    for line in corpus:
        words = line.lower().strip().split()
        # words = ['<s>'] + words + ['</s>']  # Include start and end tokens
        sent_prob = 1
        for i in range(1, len(words)):
            previous = words[i-1]
            current = words[i]
            if previous in word_index_dict and current in word_index_dict:
                bigram_prob = probs[word_index_dict[previous], word_index_dict[current]]
                sent_prob *= bigram_prob
            else:
               sent_prob *= 0  # This should not be necessary unless toy corpus is sneaky (it is not)

        # Calculate perplexity
        num_bigrams = len(words) - 1 # number of bigrams is different from the number of tokens in Q2
        perplexity = 1 / (pow(sent_prob, 1 / num_bigrams)) if sent_prob > 0 else float('inf') 
        
        # Write the perplexity to the output file
        output.write(f"{perplexity}\n")


# QUESTION 7 #####################################################################

with open('bigram_sentences', 'w') as output:
    for i in range(1,11):
        gen_text = GENERATE(word_index_dict, probs, "bigram", 15, "the")
        output.write(f"{gen_text}\n")