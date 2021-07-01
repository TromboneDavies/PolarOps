import nltk
import random
import string
import numpy as np
import pandas as pd
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.book import *
import matplotlib.pyplot as plt
from nltk import sent_tokenize
import seaborn as sns
import re
from os import listdir
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Dense
from collections import Counter



file_name = "training_data.csv"
posts = pd.read_csv(file_name, delimiter = ",")

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r', encoding='utf-8')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

def remove_punct(post):
    punctuation = string.punctuation 
    for element in punctuation:
        post = post.replace(element, "")
    post = post.replace("’", "")
    post = post.replace("—", "")
    post = post.replace("“", "")
    post = post.replace("”", "")
    return post.lower()







posts['no punc']= posts['text'].apply(remove_punct)
posts['tokens']=posts['no punc'].apply(nltk.word_tokenize)
stop = stopwords.words('english')
posts['no sw']=posts['tokens'].apply(lambda x: [item for item in x if item not in stop])
#stemmer = nltk.PorterStemmer()
#posts['stemmed'] = posts['no sw'].apply(lambda x: [stemmer.stem(y) for y in x])
posts['clean text']= posts['no sw']
#posts.drop(columns=['no punc', 'tokens', 'no sw','stemmed'], axis=1)



token_list=posts['clean text'].tolist()
# save list to file
def save_list(lines, filename):
	# convert lines to a single blob of text
	data = '\n'.join(lines)
	# open file
	file = open(filename, 'w', encoding='utf-8')
	# write text
	file.write(data)
	# close file
	file.close()
# define vocab
vocab=Counter(x for xs in token_list for x in set(xs))
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with a min occurrence
min_occurrence = 2
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')
polarized= posts.loc[posts['polarized']=='yes']
notpolarized= posts.loc[posts['polarized']=='no']
# load and clean a dataset
def load_clean_dataset(vocab, is_train):
	# load documents
	npol = (notpolarized['clean text'].tolist(), vocab, is_train)
	pol = (polarized['clean text'].tolist(), vocab, is_train)
	docs = npol + pol
	# prepare labels
	labels = array([0 for _ in range(len(npol))] + [1 for _ in range(len(pol))])
	return docs, labels




# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer


# load the vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = set(vocab.split())
# load all reviews

#train_docs, ytrain = load_clean_dataset(vocab, True)
#test_docs, ytest = load_clean_dataset(vocab, False)
# create the tokenizer
#tokenizer = create_tokenizer(train_docs)
# # encode data
# Xtrain = tokenizer.texts_to_matrix(train_docs, mode='freq')
# Xtest = tokenizer.texts_to_matrix(test_docs, mode='freq')
# print(Xtrain.shape, Xtest.shape)






