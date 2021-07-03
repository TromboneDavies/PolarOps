
# A script to generate a by-year value counts, and histogram, of a subreddit's
# submissions. To use, call plotHist() and pass a DataFrame that you loaded
# (via pd.read_csv()) from a .csv file that was produced by the collector.py
# script.

import matplotlib.pyplot as plt
import pandas as pd


def plotHist(df, subredditName):
    plt.clf()
    df = df.copy().dropna()
    if not pd.api.types.is_datetime64_any_dtype(df.date):
        df.date = df.date.astype('int').astype("datetime64[s]")
    df.date.dt.year.hist(bins=range(2008,2023))
    print(df.date.dt.year.value_counts().sort_index())
    plt.title(subredditName)
    plt.show()
