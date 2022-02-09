
# From the raw minratings.csv file, collect training data and put it in the
# same form as our summer's hand_tagged_data.csv.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import sys

# The number of "positive" (non-"other") votes a unanimous thread must receive
# from the 8 raters in order to count as usable.
MAX_VOTES_REQ = 4

pd.set_option('display.width',180)
pd.set_option('max.columns',10)

m = pd.read_csv("minratings.csv")
m = m.drop_duplicates(['comment_id','rater'])

votes_raw = pd.crosstab(m.comment_id,m.rating)
votes = votes_raw[['polarized','notpolarized']]

# Retain only unanimous threads.
votes = votes[(votes.polarized == 0) | (votes.notpolarized == 0)]

# Retain only threads with sufficient votes.
votes = votes[(votes.polarized >= MAX_VOTES_REQ) |
    (votes.notpolarized >= MAX_VOTES_REQ)]

uns = np.array([],dtype=int)
ups = np.array([],dtype=int)
ucs = np.array([],dtype=int)
btraw = pd.crosstab(m.comment_id,m.rating)
bt = btraw[["polarized","notpolarized"]]
bt = bt.value_counts().sort_index()
for p,n in [(p,n) for p in range(10) for n in range(10) if (p==0) != (n==0)]:
    if (p,n) in bt:
        uns = np.append(uns,n)
        ups = np.append(ups,p)
        ucs = np.append(ucs,bt[(p,n)])
temp = pd.DataFrame({'notpolarized':uns,'polarized':ups,'total':ucs})
usable = temp[(temp.polarized>=MAX_VOTES_REQ)|(temp.notpolarized>=MAX_VOTES_REQ)]

print(f"Old way said {usable.total.sum()} threads.")

