
# IMPORTANT!! Only run this file when you have an "old-style" (pre June 8) .csv
# file, with only one assessor's ratings in it. This will produce a new file
# called "piggied_{oldfilename.csv}" that has four columns for each assessor.
#
# Put another way, only run this file when you are the SECOND person to add
# your thread assessments to a group of threads.
#
# If you are the THIRD or FOURTH person to add your thread assessments to a
# group of threads, you should run "completeTagger.py" instead of this file.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

import os

#Returns a string of all the entire thread of a comment
def get_thread(top_level_comment, tab, final):
    first = True
    if hasattr(top_level_comment, "body"):
        lines = top_level_comment.body.split("\n")
        for line in lines:

            for i in range(tab):
                final = final + "\t"

            if first:
                final = final + ">"
                first = False

            final = final + line + "\n"

        for comment in top_level_comment._replies:
            final = get_thread(comment, tab + 1, final)

    return final

#Ask user questions to prepare for hand tagging
existing_file = input("What is the name of the existing data file? " +
    "(Hint: it should NOT begin with \"piggied_\".)\n")
while not os.path.isfile(existing_file):
    existing_file = input("No such file! Try again:\n")
preexisting = pd.read_csv(existing_file, header=None)
old_assessor = preexisting.iloc[0][5]
preexisting.columns = ['Subreddit','SubmissionID','CommentID','text','ttype',
    'assessor','date']

user = input("\nType 1/2/3/4 after each thread to designate which type it is.\n\nYou can type \"exit\" to stop the program.\n\nPlease type your name to begin.\n") 

assessors = ["Stephen","TJ","Alexis","Veronica"]
tt = ""
legit = ["1", "2", "3", "4"]
comma = ","

subreddits = preexisting.Subreddit.copy()
submissionIDs = preexisting.SubmissionID.copy()
commentIDs = preexisting.CommentID.copy()
texts = preexisting.text.copy()
ttypes = { name : np.repeat(-99,len(preexisting)) for name in assessors }
ttypes[old_assessor] = preexisting.ttype.copy()
dates = preexisting.date.copy()


for i,thread in enumerate(preexisting.itertuples()):
    print(thread.text.replace("\\n","\n"))
    tt = input("\n=============================================\n" +
        "What thread type is thread {}/{}, {}? (1,2,3,4,exit) ".
        format(i+1,len(preexisting),user))
    if tt == "exit":
        break

    subreddits[i] = preexisting.iloc[i].Subreddit
    submissionIDs[i] = preexisting.iloc[i].SubmissionID
    commentIDs[i] = preexisting.iloc[i].CommentID
    texts[i] = preexisting.iloc[i].text
    ttypes[user][i] = tt
    dates[i] = preexisting.iloc[i].date
 
columns = { 'Subreddit': subreddits,
    'SubmissionID': submissionIDs,
    'CommentID': commentIDs,
    'text': texts,
    'date': dates }
columns.update(ttypes)

newguy = pd.DataFrame(columns)
newguy.to_csv("piggied_{}".format(existing_file), index=None)
print("Wrote output to piggied_{}!".format(existing_file))

