import nltk
from nltk import FreqDist, pos_tag
from nltk.corpus import brown 
from collections import Counter
import matplotlib.pyplot as plt

# import the full brown corpus (uncomment if first time loading)
# nltk.download('brown')
#nltk.download('averaged_perceptron_tagger')

# check which genres are available: we choose genre of 'hobbies' and 'humor'
print(brown.categories())



# extract all the 
def desc_freq(categories = None):
    if categories:
        words = brown.words(categories=categories)
        sentences = brown.sents(categories=categories)

    else:
        words = brown.words()
        sentences = brown.sents()

        
    words = [w.lower() for w in words if w.isalpha()]

    word_freq = FreqDist(w for w in words)
    desc_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    num_tokens = sum(len(w) for w in words)
    num_types = len(set(words))
    num_words = len(words)
    avg_words_per_sentence = sum(len(sentence) for sentence in sentences) / len(sentences)
    avg_word_length = sum(len(word) for word in words) / num_words
    
    # POS
    pos_tags = pos_tag(words)
    pos_freq = FreqDist(tag for (word, tag) in pos_tags)
    most_common_tags = pos_freq.most_common(10)
    
    inf_dict = {
        'Sorted Frequency': desc_freq[:10],
        'Number of tokens: ': num_tokens,
        'Number of types: ': num_types,
        'Number of words': num_words,
        'Average number of words per sentence:': avg_words_per_sentence,
        'Average word length': avg_word_length,
        'Most frequent POS tags': most_common_tags
    }
    
    return inf_dict, desc_freq

def print_(inf_dict):
    for key, value in inf_dict.items():
        print(f"{key} {value}.")

def plot(desc_freq, title):
    frequencies = [freq for _, freq in desc_freq]

    # Linear plot
    plt.subplot(1, 2, 1)
    plt.plot(frequencies, marker='o', linestyle='-')
    plt.title(f'Linear {title}')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')

    # Log-log plot
    plt.subplot(1, 2, 2)
    plt.loglog(frequencies, marker='o', linestyle='-')
    plt.title(f'Log-Log {title}')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# (i) the whole corpus
corps_inf, corpus_desc_freq = desc_freq()
print("(i) full corpus: ")
print_(corps_inf)
plot(corpus_desc_freq, 'full corpus')

print()

# (ii) 2 categories of corpus
words, categories = ['hobbies', 'humor']
categories_inf, categories_desc_freq = desc_freq(categories = categories)
print("(ii) two categories: ")
print_(categories_inf)
plot(categories_desc_freq, 'hobbies and humor')

