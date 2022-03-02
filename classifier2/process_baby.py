
# Let's see how our predictions do against the training data (we would think
# this would be high.) This data is in TheHandTaggedDataBaby.csv. So first, run
# process.py with baby.csv as an argument to produce baby_predictions. Then run
# this script.

import pandas as pd
import matplotlib.pyplot as plt


baby = pd.read_csv("baby.csv", comment="#")
baby_pred = pd.read_csv("baby_predictions.csv", comment="#")
confusion = pd.merge(baby, baby_pred, left_on="comment_id",
    right_on="comment_ids")[['comment_id','groundTruth','prediction','text']]

confusion['prediction'] = confusion['prediction'] == 'polar'

print(pd.crosstab(confusion.groundTruth, confusion.prediction))
