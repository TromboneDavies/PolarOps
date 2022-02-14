
import numpy as np
import pandas as pd
import sqlite3
from os.path import exists

summer_ratings = pd.read_csv("classifier/summer.csv")
summer_data = pd.read_csv("classifier/hand_tagged_data.csv")
fall_ratings = pd.read_csv("minions/fall.csv")
fall_data_conn = sqlite3.connect("minions/fall2021.sqlite")

# Pending email to F4 about what to do with summer dups
summer = pd.merge(summer_ratings,summer_data,on="comment_id")[['comment_id',
    'polarized_x','text']]
summer.columns = ['comment_id','polarized','text']

fall_text = np.empty(len(fall_ratings),dtype=object)
num = 0
for comment_id in fall_ratings.comment_id:
    for pile in range(1,4):
        results = fall_data_conn.execute(f"""
            select text from pile{pile} where comment_id=?
        """, (comment_id,)).fetchone()
        if results:
            fall_text[num] = results[0]
            num += 1
fall = pd.DataFrame({'comment_id':fall_ratings.comment_id,
    'polarized':fall_ratings.polarized,'text':fall_text})

both = pd.concat([summer,fall])

preamble = \
"""# The entire hand-tagged data set: 158 threads from summer (F4) plus 370
# threads from fall (Minions). Each thread has had at least 4 "positive"
# consensus votes (i.e., for either polarized or not-polarized) and no
# dissenting votes (i.e., there could have been "other/don't-know" votes, but
# no votes for the opposite polarized/not-polarized).
"""

if exists("TheHandTaggedDataBaby.csv"):
    print("TheHandTaggedDataBaby.csv already exists! Cowardlyly refusing to " +
        "overwrite.")
else:
    with open("TheHandTaggedDataBaby.csv","w") as f:
        f.write(preamble)
        both.to_csv(f,index=False)

#"TheHandTaggedDataBaby.csv"
