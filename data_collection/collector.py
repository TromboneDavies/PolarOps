import praw
import datetime as dt
import os.path
from os import path
from psaw import PushshiftAPI
import csv

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
sub = input("What is the name of the subreddit you are collecting (all lower case)?\n")
n = int(input("What is your limit?\n"))
date = input("What year would you like to pull from?\n")

start_epoch = int(dt.datetime(int(date), 1, 1).timestamp())
subreddit = r.subreddit(sub)

#CSV file
header = ['subreddit','submission_id','comment_id','text','date','batch_num']
if not path.exists(name):
    new = True
with open(name, 'a') as f:
    if new:
        f.write(comma.join(header) + "\n")
    else:
        with open(name, 'r') as t:
            submission = r.submission(t.readlines()[-1].split(",")[0])
            start_epoch = int(submission.created_utc)
        
    while True:
        batch_num = batch_num + 1
        print("The new start_epoch for the query is: {}".format(start_epoch+1))
        remember_this = start_epoch
        posts =  list(api.search_submissions(after=start_epoch+1,
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
            start_epoch = int(post.created_utc)
            if start_epoch > remember_this:
                print("Yep, carry on...")
            else:
                print("WHOA: just read {}, which is earlier than {}!".
                    format(start_epoch, remember_this))
    f.close()
