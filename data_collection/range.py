# Script to print human-readable information about how many threads are in
# various months of a .csv file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import sys
from datestuff import makeDateReadable
from calendar import month_abbr

if len(sys.argv) == 1:
    df_name = input("What CSV file would you like the range of? ")
else:
    df_name = sys.argv[1]

if not df_name.endswith(".csv"):
    df_name = df_name + ".csv"

df = pd.read_csv(df_name).dropna()

dtinfo = df.date.astype(int).astype("datetime64[s]")
df['year'] = dtinfo.dt.year.astype(int)
df['month_num'] = dtinfo.dt.month.astype(int)

counts = df.groupby(['year','month_num']).comment_id.count()
sorted_counts = counts.sort_index(ascending=False)
thing = [(y,month_abbr[m]) for y,m in sorted_counts.index]
sorted_counts.index=pd.MultiIndex.from_tuples(thing)
with pd.option_context('display.max_rows', None):
    print(sorted_counts)

print("There are a total of {} entries, from {} to {}.".format(len(df),
    makeDateReadable(df.date.min()), makeDateReadable(df.date.max())))
