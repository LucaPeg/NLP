#!/usr/bin/env python3
import pandas as pd


"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

# TODO: read brown_vocab_100.txt into word_index_dict

# TODO: write word_index_dict to word_to_index_100.txt

# import the txt

filename = '/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/brown_vocab_100.txt'
output_filename = '/Users/noalassally/Documents/master /Jaar 2/NLP/A2_2024/word_to_index_100.txt'
word_index_dict = {}

# read brown_vocab_100.txt into word_index_dict
def write_to_dict(filename, word_index_dict):
    with open(filename, 'r') as file:
        for index, line in enumerate(file):
            word = line.strip()  
            word_index_dict[word] = index
    return word_index_dict

# create word index dict
write_to_dict(filename, word_index_dict)
    
# write word_index_dict to word_to_index_100.txt
with open(output_filename, 'w') as wf:
    wf.write(str(word_index_dict))


print(word_index_dict['all'])
print(word_index_dict['resolution'])
print(len(word_index_dict))
