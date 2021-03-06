# Perform initial exploratory analysis of hand-tagged data.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from loadData import ht, vectorize
from sklearn.feature_selection import SelectKBest, f_classif

# See also https://scikit-learn.org/stable/modules/classes.html#text-feature-extraction-ref
from sklearn.feature_extraction.text import CountVectorizer

# Show bar graph of N most common words.
N = 30
features, meanings, _ = vectorize(ht.text)

plt.figure()
plt.rc('xtick',labelsize=6)
all_words = list(meanings)
all_counts = features.sum(axis=0).tolist()
all_counts, all_words = zip(*[(c,n) for c,n in sorted(
    zip(all_counts,all_words), reverse=True)])
plt.bar(np.arange(N), list(all_counts)[:N], color="brown")
plt.xticks(np.arange(N), all_words[:N], rotation=60)
plt.title("Most common non-stop-word unigrams")
plt.tight_layout()
plt.savefig("mostCommonWords.png", dpi=300)


# Show histogram of thread lengths.
plt.figure()
plt.title("Thread length")
lengths = pd.Series([ len(ct) for ct in ht['text'] ])
lengths.hist(bins=30)
plt.xlabel("# of characters")
plt.axvline(x=lengths.mean(), color="black", linestyle="dashed")
plt.axvline(x=lengths.median(), color="red", linestyle="dashed")
plt.annotate(f"mean: {lengths.mean():.1f} chars", (2000,360), color="black")
plt.annotate(f"median: {lengths.median():.1f} chars", (3000,400), color="red")
plt.savefig("threadLengths.png", dpi=300)


# According to the Google Text Classification Guide, a good rule of thumb is to
# look at the ratio of number of samples to the number of words per sample. If
# < 1500, an n-gram model is better than a sequence model. Ours is laughably
# smaller than this (much smaller than 1, actually!) so n-grams it is.
# https://developers.google.com/machine-learning/guides/text-classification/step-2-5
ratio = len(ht) / lengths.sum()
print(f"The Google ratio is {ratio:.5f}.")


# Compute and print the most indicative features. 
NUM_FEATURES = 50
features, feature_names, vocab = vectorize(ht.text)

selector = SelectKBest(f_classif, k='all')
selector.fit(features, ht.polarized)
feature_names_for_df = selector.get_feature_names_out(feature_names)
scores_for_df = selector.scores_
important_features = pd.DataFrame({'feature_name':feature_names_for_df,
    'score':scores_for_df}).set_index('feature_name')
important_features = important_features.sort_values("score",
    ascending=False)[0:NUM_FEATURES]


from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(features, ht.polarized)

indications = np.empty(NUM_FEATURES, dtype="object")
for i, feature_name in enumerate(important_features.index[0:NUM_FEATURES]):

    feature_index = np.where(feature_names == feature_name)[0][0]

    # Note this is counting # of occurrences, not just presence/absence...
    num_polar = sum(features[ht.polarized][:, feature_index])
    num_nonpolar = sum(features[~ht.polarized][:, feature_index])

    prop_polar = num_polar / sum(ht.polarized)
    prop_nonpolar = num_nonpolar / sum(~ht.polarized)

    if prop_polar > prop_nonpolar:
        indications[i] = "polar"
    elif prop_polar < prop_nonpolar:
        indications[i] = "nonpolar"
    else:
        indications[i] = "tie"

important_features['indication'] = indications

print(f"The {NUM_FEATURES} most indicative features are:")
important_features = important_features.sort_values("score",
    ascending=False)[0:NUM_FEATURES]
print(important_features)
