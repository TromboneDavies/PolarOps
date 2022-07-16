# Perform grid search for best parameters on count -> TFIDF -> either
# SGDlinearSVM or MultinomialNB.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib    # to save/restore models
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

model = "SGD/SVM"   # or "MultinomialNB"
K = 10  # number of folds in cross-validated grid search

np.set_printoptions(precision=3, suppress=True)

threads = pd.read_csv("TheHandTaggedDataBaby.csv", comment="#")
print(f"Read {len(threads)} threads.")

classifier = make_pipeline(
    CountVectorizer(encoding="utf-8",lowercase=True),
    TfidfTransformer(),
    MultinomialNB() if model == "MultinomialNB" else
    SGDClassifier(loss='hinge',  # linear SVM classifier
        penalty='l2',
        random_state=13,
        max_iter=5,
        tol=None)
)

parameters = {
    'countvectorizer__ngram_range': [(1,1),(1,2)],
    'countvectorizer__stop_words': [None], # empirically, 'english' sucks
    'countvectorizer__strip_accents': [None], # empirically, no diff
    'countvectorizer__min_df': [0.0],   # empirically, >= .1 is bad
    'countvectorizer__max_df': [0.8,0.9,1.0],
    'countvectorizer__binary': [True,False],
    'tfidftransformer__use_idf': [True,False],
    'multinomialnb__alpha' if model=="MultinomialNB" else
    'sgdclassifier__alpha': np.logspace(-4,-3,5)
}

gs = GridSearchCV(classifier, parameters, cv=K, n_jobs=-1, verbose=1)
gs.fit(threads.text, threads.polarized)

print(f"Best score: {100*gs.best_score_:.1f}")
print(f"Best params: {gs.best_params_}")
results = pd.DataFrame(gs.cv_results_)[['param_countvectorizer__ngram_range',
    'param_countvectorizer__stop_words',
    'param_countvectorizer__strip_accents',
    'param_countvectorizer__min_df',
    'param_countvectorizer__max_df',
    'param_countvectorizer__binary',
    'param_tfidftransformer__use_idf',
    'param_multinomialnb__alpha' if model=="MultinomialNB" else
    'param_sgdclassifier__alpha',
    'mean_test_score']]
results.columns = [ col[col.find("__")+2:] if col.startswith("param") else col
    for col in results.columns ]

pd.set_option('display.width',200)
pd.set_option('display.max_columns',20)
print("Best 10 param combos:")
print(results.sort_values('mean_test_score',ascending=False)[0:10])

print("Saving best classifier in skpolarops.joblib...")
joblib.dump(gs.best_estimator_, 'skpolarops.joblib')

# Takeaways:
# SGD with linear SVM is slightly better than Multinomial NB with ~0
#   smoothing (~80% vs ~78%)
# We definitely want:
# - unigrams only
# - either no stop words or a better stop words list than default "english"
# - min_df of 0
# - TF/IDF (not just TF)
# - a pretty low SGD alpha
# Don't seem to make much difference:
# - max_df
# - binarizing

