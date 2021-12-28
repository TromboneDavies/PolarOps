
Classifier:
-----------

Probably main choices:

- Use unigrams + bigrams.

- Vectorize using TF/IDF

- 80/20 split for validation

- Skip tokens that appear fewer than _k_ (=2?) times, then use top 20,000
  features ("top" = feature importance score, using `f_classif` from `sklearn.feature_selection`)

Other things to compare:

- just unigrams; just bigrams

- one-hot or count vectorization instead of TF/IDF 

- word embeddings (compute centroid of each document, try with and without
weighting by word frequency -- see 7/2/21 email Jim Martin)
