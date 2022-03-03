
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from os.path import exists
from glob import glob


national_subreddits = ['politics', 'conservative', 'progressive', 'democrats',
'republican', 'centrist', 'liberal', 'PoliticalDebate', 'moderatepolitics',
'firstamendment', 'secondamendment', 'DebateaCommunist', 'progun',
'bannedfromthe_donald', 'Libertarian']

state_subreddits = [ 'ArizonaPolitics', 'illinoispolitics',
'WisconsinPolitics', 'SouthCarolinaPolitics', 'MinnesotaPolitics',
'VirginiaPolitics', 'NCPolitics', 'TexasPolitics', 'NYSPolitics',
'MarylandPolitics' ]

print("Reading all_predictions.csv...", flush=True)
all_pred = pd.read_csv("all_predictions.csv")

if not exists("all_polar_scores.png"):
    print("Creating all_polar_scores.png...", flush=True)
    plt.clf()
    plt.scatter(all_pred.year + all_pred.month/12, all_pred.polar_score,
        marker=".", alpha=.01)
    plt.savefig("all_polar_scores.png")
else:
    print("(Skipping all_polar_scores.png.)")

if not exists("date_hist.png"):
    print("Creating date_hist.png...", flush=True)
    plt.clf()
    num_bins = (all_pred.year.max() - all_pred.year.min() - 1)*12 + \
        12 - all_pred[all_pred.year==min(all_pred.year)].month.min() + 1 + \
        all_pred[all_pred.year==max(all_pred.year)].month.max()
    num_bins = int(num_bins / 2)   # One bin for each two months
    pd.Series(all_pred.year + all_pred.month/12).hist(bins=int(num_bins/2),
        density=True, figsize=(8,5))
    plt.title("Dates of all sucked threads")
    plt.xlabel("Year/month")
    
    plt.savefig("date_hist.png")
else:
    print("(Skipping date_hist.png.)")

if not exists("mythicalMeanScore.png"):
    print("Creating mythicalMeanScore.png...", flush=True)
    plt.clf()
    all_pred['date'] = all_pred.year + all_pred.month/12
    grouped = all_pred.groupby(['date']).polar_score.mean()
    plt.plot(grouped.index, grouped)
    plt.title("The Mythical Graph (mean polarization score)")
    plt.xlabel("Year/month")
    plt.ylabel("Mean polarization score of all threads")
    plt.ylim((0,.5))
    plt.savefig("mythicalMeanScore.png")
else:
    print("(Skipping mythicalMeanScore.png.)")

if not exists("mythicalPercentPolarized.png"):
    print("Creating mythicalPercentPolarized.png...", flush=True)
    plt.clf()
    all_pred['date'] = all_pred.year + all_pred.month/12
    all_pred['pred'] = (all_pred.polar_score > .5) * 100.0
    grouped = all_pred.groupby(['date']).pred.mean()
    plt.plot(grouped.index, grouped)
    plt.title("The Mythical Graph (% threads polarized)")
    plt.xlabel("Year/month")
    plt.ylabel("Percentage of all threads polarized")
    plt.ylim((0,50))
    plt.savefig("mythicalPercentPolarized.png")
else:
    print("(Skipping mythicalPercentPolarized.png.)")


# Dividing subreddit plots
if len(glob("mythicalBySubreddit*.png")) < 5:
    print("Creating mythicalBySubreddit.png...", flush=True)
    plt.clf()
    all_pred['date'] = all_pred.year + all_pred.month/12
    all_pred['pred'] = (all_pred.polar_score > .5) * 100.0

    # There has to be a better way to do this...
    grouped = pd.DataFrame(all_pred.groupby(['date','subreddit']).pred.mean())
    indexed = grouped.index.to_frame()
    indexed["pred"] = grouped.pred
    indexed.index = np.arange(len(indexed))

    indexed.set_index('date', inplace=True)
    thirdway_index = int(len(national_subreddits)/3)
    twothirdsway_index = thirdway_index * 2
    nationalA = indexed[indexed.subreddit.isin(
        national_subreddits[:thirdway_index])]
    nationalB = indexed[indexed.subreddit.isin(
        national_subreddits[thirdway_index:twothirdsway_index])]
    nationalC = indexed[indexed.subreddit.isin(
        national_subreddits[twothirdsway_index:])]
    halfway_index = int(len(state_subreddits)/2)
    stateA = indexed[indexed.subreddit.isin(
        state_subreddits[:halfway_index])]
    stateB = indexed[indexed.subreddit.isin(
        state_subreddits[halfway_index:])]

    for group in [ f"national{g}" for g in "ABC" ] + \
            [ f"state{g}" for g in "AB" ]:
        plt.clf()
        globals()[group].groupby('subreddit')['pred'].plot(
            legend=True)
        plt.title("The Mythical Graph by Subreddit (% threads polarized)")
        plt.xlabel("Year/month")
        plt.ylabel("Percentage of all threads in Subreddit polarized")
        plt.ylim((0,100))
        plt.xlim((2007,2022))
        plt.savefig(f"mythicalBySubreddit{group}.png")

else:
    print("(Skipping mythicalBySubreddit.png.)")

