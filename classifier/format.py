import pandas as pd
from os.path import exists

# Reformat our summer data to match our fall data: a single Series, with
# comment_id as the index and rating (True=polarized, False=notpolarized) as
# the value.
htd = pd.read_csv("hand_tagged_data.csv").set_index('comment_id')
if exists("summer.csv"):
    print("summer.csv already exists! Cowardlyly refusing to overwrite.")
else:
    (htd.polarized=="yes").to_csv("summer.csv")
