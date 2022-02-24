# Given a model stored in the PolarOps.model directory structure, and the
# vocabulary (as a dictionary of occurrences) in the file vocab.json, predict
# the polarization status of some threads.
#
# If perchance there is no PolarOps.model directory and/or vocab.json, you'll
# want to run the enchilada.py script with command-line arg train_frac=1.

import tensorflow as tf
import json
import pickle
from sklearn.feature_selection import SelectKBest, f_classif
from enchilada import vectorize, clean


# Load the saved model (enchilada.py did that).
model = tf.keras.models.load_model("PolarOps.model")

# Load the vocabulary because we can't vectorize new texts without it.
with open("vocab.json","r") as f:
    vocab = json.load(f)


# Make a couple "obviously" polar/nonpolar examples.
polarized = "This is a fucking trump hating maga shit text"
nonpolarized = """
The government establishes a market. That's what it means. It's only an issue
for individuals.
"""

# Clean the text (really does nothing but remove URLs.)
polarized = clean(polarized)
nonpolarized = clean(nonpolarized)

# Vectorize these (cleaned) texts. The first argument is a list of cleaned
# texts. (None means we're not using validation/test data.) The third argument
# is our vocab list with counts.)
vecs, _, _ = vectorize([polarized, nonpolarized], None, vocab)

# Load the "selector" which is really the thing that can vectorize texts (i.e.,
# turn English text into arrays of 2000 numbers.)
with open("selector.pickle","rb") as f:
    selector = pickle.load(f)
trans_vecs = selector.transform(vecs).astype('float32')

print(model.predict(trans_vecs))
