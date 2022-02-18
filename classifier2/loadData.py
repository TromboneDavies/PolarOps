
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# See also https://scikit-learn.org/stable/modules/classes.html#text-feature-extraction-ref
from sklearn.feature_extraction.text import CountVectorizer


random.seed(2022)

TRAIN_FRAC = .8



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
        stop_words='english',     # could use 'english'
        ngram_range=(1,1),
        analyzer='word',
        min_df=0.0,
        max_df=1.0,
        max_features=None,   # unlimited
        vocabulary=None,     # use vocab in the texts
    )
    return (vectorizer.fit_transform(texts).toarray(),
        vectorizer.get_feature_names_out())


ht = pd.read_csv("TheHandTaggedDataBaby.csv", comment="#")

# Eliminate the top 10% longest threads, since they comprise most of a long,
# probably-not-representative tail. (See email chain 2/18/2022.)
ht = ht[ht.text.str.len() < ht.text.str.len().quantile(.9)]

cleaned_texts = np.empty(len(ht), dtype=object)
for i,row in enumerate(ht.itertuples()):
    cleaned_texts[i] = clean(row.text)
ht['text'] = cleaned_texts

train = ht.sample(frac=TRAIN_FRAC)
validate = ht[~ht.index.isin(train.index)]


