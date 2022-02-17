
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import loadData

# See also https://scikit-learn.org/stable/modules/classes.html#text-feature-extraction-ref
from sklearn.feature_extraction.text import CountVectorizer

ht = pd.read_csv("TheHandTaggedDataBaby.csv", comment="#")
cleaned_texts = np.empty(len(ht), dtype=object)
for i,row in enumerate(ht.itertuples()):
    cleaned_texts[i] = loadData.clean(row.text)
ht['text'] = cleaned_texts

# Show bar graph of N most common words.
N = 30
features, meanings = loadData.vectorize(ht.text)

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

