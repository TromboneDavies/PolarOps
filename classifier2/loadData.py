# Load the data from TheHandTaggedDataBaby.csv and prepare for classification:
#   - Shuffle data
#   - Split into training and validation sets
#   - Clean text (case, punctuation, etc.; see clean() function and options to
#     CountVectorizer constructor in vectorize() function)
#   - Vectorize according to counts (later: TF/IDF)
#
# The following variables are available after running this file:
#   - ht: the entire shuffled and labeled hand-tagged data set
#   - train: the subset of ht used for training
#   - validate: the fraction of ht used for validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# See also https://scikit-learn.org/stable/modules/classes.html#text-feature-extraction-ref
from sklearn.feature_extraction.text import CountVectorizer


random.seed(2022)

TRAIN_FRAC = .8     # Fraction of data used for training (not validation)
NGRAM_MAX = 1       # 1 - unigrams only; 2- unigrams+bigrams; etc.



def clean(dirty):
    """Given a real-live Reddit comment thread, prepare it for processing by
    converting to lower-case, removing punctuation, etc.
    """
    return dirty.replace("'","")

def vectorize(texts):
    """Given a list of (cleaned) texts, turn it into a matrix of features. This
    involves both the traditional "tokenization" step and occurrence
    counting."""
    vectorizer = CountVectorizer(
        encoding='utf-8',
        lowercase=True,
        stop_words='english',     # could use None
        ngram_range=(1,NGRAM_MAX),
        analyzer='word',
        min_df=0.0,
        max_df=1.0,
        max_features=None,   # unlimited
        vocabulary=None,     # use vocab in the texts
    )
    return (vectorizer.fit_transform(texts).toarray(),
        vectorizer.get_feature_names_out())


ht = pd.read_csv("TheHandTaggedDataBaby.csv", comment="#")
ht.sample(frac=1)   # Shuffle the data before doing anything else.

# Eliminate the top 10% longest threads, since they comprise most of a long,
# probably-not-representative tail. (See email chain 2/18/2022.)
ht = ht[ht.text.str.len() < ht.text.str.len().quantile(.9)]

cleaned_texts = np.empty(len(ht), dtype=object)
for i,row in enumerate(ht.itertuples()):
    cleaned_texts[i] = clean(row.text)
ht['text'] = cleaned_texts

train = ht.sample(frac=TRAIN_FRAC)
validate = ht[~ht.index.isin(train.index)]


