# Given a model stored in the PolarOps.model directory structure, and the
# vocabulary (as a dictionary of occurrences) in the file vocab.json, predict
# the polarization status of some threads.
#
# If perchance there is no PolarOps.model directory and/or vocab.json, you'll
# want to run the enchilada.py script with command-line arg train_frac=1.

import tensorflow as tf
import sys
import json
import pickle
import datetime
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

# Open the CSV file of threads.
if len(sys.argv) < 2:
    sys.exit("Usage: process.py file_with_threads.csv.")
filename = sys.argv[1]

print(f"Loading {filename}...")
threadsdf = pd.read_csv(filename)

total_num_threads = len(threadsdf)
print(f"Counted {total_num_threads} threads.")

print(f"Converting times to month/year...")
dtinfo = threadsdf.date.astype(int).astype("datetime64[s]")
threadsdf['year'] = dtinfo.dt.year.astype(int)
threadsdf['month_num'] = dtinfo.dt.month.astype(int)

# Initialize columns of to-be-created DataFrame
print(f"Initializing columns...")
predictions = np.empty(total_num_threads,dtype=object)
polar_scores = np.empty(total_num_threads,dtype=float)


for i,row in enumerate(threadsdf.itertuples()):
    if i % 10 == 0:
        print(f"  Processing thread #{i} of {total_num_threads} "
            f"({i/total_num_threads*100:.1f}%)...")
    # Get the vector of the text
    vec, _, _ = vectorize([clean(row.text)], None, vocab)
    # Transform the vector into a bunch of float values to give to the neural
    # net
    trans_vec = selector.transform(vec).astype('float32')
    # Get the score based on the neural nets analysis
    polar_score = model.predict(trans_vec)
    
    if polar_score >= .5:
        predictions[i] = "polar"
    else:
        predictions[i] = "nonpolar"
    polar_scores[i] = polar_score
    
# Create new DataFrame to store the predictions stuff to graph
prediction_df = pd.DataFrame({
    'year':threadsdf.year,
    'month':threadsdf.month_num,
    'subreddit':threadsdf.subreddit,
    'comment_ids':threadsdf.comment_id,
    'prediction':predictions,
    'polar_score':polar_scores
})

# Output the predictions DataFrame to a csv file
prediction_df.to_csv(filename[:-4] + "_predictions.csv", index=None)

