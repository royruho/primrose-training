import nltk
import pandas as pd
corpus = open(r'C:\Users\royru\AppData\Roaming\nltk_data\corpora\gutenberg\austen-persuasion.txt')
text = corpus.read()
corpus.close

from nltk import word_tokenize
text_tokens = word_tokenize(text)
# remove all tokens that are not alphabetic
words_only = [word for word in text_tokens if word.isalpha()]
# number of words in text
print (len(words_only))
# number of distinct words in text
from nltk import FreqDist
fdist = FreqDist()
for word in words_only:
    fdist[word.lower()] +=1
print (len(fdist))