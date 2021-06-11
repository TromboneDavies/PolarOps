
# IMPORTANT!! Read the comments at the top of piggyBackTagger.py before running
# this file!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns

import sys
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

assessors = ["Stephen","TJ","Alexis","Veronica"]

#Ask user questions to prepare for hand tagging
existing_file = input("What is the name of the existing data file? " +
    "(Hint: it should start with \"piggied_\".)\n")
while not os.path.isfile(existing_file):
    existing_file = input("No such file! Try again:\n")
preexisting = pd.read_csv(existing_file)
if not all([ a in preexisting.columns for a in assessors ]):
    print("Your {} file is jacked up! Quitting now.".format(existing_file))
    sys.exit(1)

user = input("\nType 1/2/3/4 after each thread to designate which type it is.\n\nYou can type \"exit\" to stop the program.\n\nPlease type your name to begin.\n") 

tt = ""
legit = ["1", "2", "3", "4"]
comma = ","

new_assessments = np.repeat(-99,len(preexisting))

for i,thread in enumerate(preexisting.itertuples()):
    print(thread.text.replace("\\n","\n"))
    if preexisting.iloc[i][user] != -99:
        print("You already assessed this as {}.".format(preexisting.iloc[i][user]))
        new_assessments[i] = preexisting.iloc[i][user]
    else:
        tt = input("\n=============================================\n" +
            "What thread type is thread {}/{}, {}? (1,2,3,4,exit) ".
            format(i+1,len(preexisting),user))
        if tt == "exit":
            break
        new_assessments[i] = tt

 
preexisting[user] = new_assessments

preexisting.to_csv(existing_file, index=None)
print("Wrote output to {}!".format(existing_file))

