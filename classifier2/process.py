# Given a model stored in the PolarOps.model directory structure, and the
# vocabulary (as a dictionary of occurrences) in the file vocab.json, predict
# the polarization status of some threads.
#
# If perchance there is no PolarOps.model directory and/or vocab.json, you'll
# want to run the enchilada.py script with command-line arg train_frac=1.

import tensorflow as tf
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from enchilada import vectorize, clean


# Load the saved model (enchilada.py did that).
model = tf.keras.models.load_model("PolarOps.model")

# Load the vocabulary because we can't vectorize new texts without it.
with open("vocab.json","r") as f:
    vocab = json.load(f)

# Load the pickle 
with open("selector.pickle","rb") as f:
    selector = pickle.load(f)

# Initialize columns
dates = np.array([])
subreddits = np.array([])
comment_ids = np.array([])
predictions = np.array([])
polar_scores = np.array([])

# Open the file
filename = sys.argv[1]
thefile = pd.read_csv(filename)

for row in thefile.itertuples():
    # Get the vector of the text
    vec, _, _ = vectorize([clean(row.text)], None, vocab)
    # Transform the vector into a bunch of float values to give to the neural
    # net
    trans_vec = selector.transform(vec).astype('float32')
    # Get the score based on the neural nets analysis
    polar_score = model.predict(trans_vec)
    
    # Add this rows information to the columns
    dates = np.append(dates, row.date)
    subreddits = np.append(subreddits, row.subreddit)
    comment_ids = np.append(comment_ids, row.comment_id)
    if polar_score >= .5:
        predictions = np.append(predictions, "polar")
    else:
        predictions = np.append(predictions, "not polar")
    polar_scores = np.append(polar_scores, polar_score)
    
# Create new dataframe to store the stuff being graphed
tjs_df = pd.DataFrame({ 'date':dates, 'subreddit':subreddits,
'comment_ids':comment_ids, 'predictions':predictions, 'polar_scores':
polar_scores })
# Output the dataframe to a csv file
tjs_df.to_csv(filename[:-4] + "GraphReady.csv")

