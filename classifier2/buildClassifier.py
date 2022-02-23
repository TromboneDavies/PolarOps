# Build a classifier.
# TODO: word embeddings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.feature_selection import SelectKBest, f_classif
from loadData import ht, train, validate, vectorize


# The number of "best" features to use, as measured by f_classif.
NUM_TOP_FEATURES = 100

# TODO: should we be vectorizing only train, not all of ht?
train_vecs, validate_vecs, feature_names, _ = \
    vectorize(train.text, validate.text)

selector = SelectKBest(f_classif, k=NUM_TOP_FEATURES)
selector.fit(train_vecs, train.polarized)
x_train = selector.transform(train_vecs).astype('float32')
x_validate = selector.transform(validate_vecs).astype('float32')
fns = selector.get_feature_names_out(feature_names)
