import praw
import datetime as dt
import os.path
from os import path
from psaw import PushshiftAPI
import pandas as pd
import csv

def makeDateReadable(timestamp, longform=False):
    if not longform:
        return dt.datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M%p")
    else:
        return dt.datetime.fromtimestamp(timestamp).strftime("%m/%d/%y %I:%M:%S%p")

#Returns a string of the entire thread of a comment
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

r = praw.Reddit("sensor1")
api = PushshiftAPI(r)
comma = ","
new = False
batch_num = 0

#Questions for collection
name = input("What is the name of the file you want your data to go in?\n")
if not name.endswith('.csv'):
    name += '.csv'
sub = input("What is the name of the subreddit you are collecting (all lower case)?\n")
n = int(input("What is your batch size?\n"))

subreddit = r.subreddit(sub)

#CSV file
header = ['subreddit','submission_id','comment_id','text','date','batch_num']
if not path.exists(name):
    new = True
with open(name, 'a') as f:
    if new:
        print("No file named {} yet...we'll create a new one.".format(name))
        year = int(input("What year would you like to pull from?\n")) + 1
        end_epoch = int(dt.datetime(int(year), 1, 1).timestamp())
        print("We'll begin at {} and go backwards....".format(
            makeDateReadable(end_epoch-1)))
        f.write(comma.join(header) + "\n")
    else:
        with open(name, 'r') as t:
            print("We'll add to {}.".format(name))
            existing_df = pd.read_csv(name)
            end_epoch = int(existing_df.date.min() - 1)
            batch_num = existing_df.batch_num.max()
            print("We'll resume at {} and go backwards....".format(
                makeDateReadable(end_epoch-1)))
        
    while True:
        batch_num = batch_num + 1
        print("Starting new batch {} at time {}...".format(batch_num,
            makeDateReadable(end_epoch-1,True)))
        remember_this = end_epoch
        posts =  list(api.search_submissions(before=end_epoch-1,
                    subreddit=sub, limit=n))

        #Loop through posts and put their threads in a csv file
        for post in posts:
            for top_level_comment in r.submission(post).comments:
                if hasattr(top_level_comment, "body"):
                    words = get_thread(top_level_comment, 0, "").replace('"', "'")
                    words = words.replace("\n", "\\n")
                    write = [sub, top_level_comment.link_id,
                    top_level_comment.id, '"' + words + '"',
                    str(top_level_comment.created_utc), str(batch_num)]
                    f.write(comma.join(write) + "\n")
                    f.flush()
                    print(" {}".format(
                        makeDateReadable(top_level_comment.created_utc,True)))
            end_epoch = int(post.created_utc)
    f.close()
