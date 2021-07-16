import praw
from datetime import datetime
import datetime as dt
import os.path
from os import path
from psaw import PushshiftAPI
import pandas as pd
import csv
import sys
from datestuff import makeDateReadable, calculatePrevMonth

header = ['subreddit','submission_id','comment_id','text','date','batch_num']

# Make this number (way) higher if you don't want it to "be monthly."
THREADS_PER_MONTH = 999999999999999

# To create a new .py file, run with four arguments:
#     collector.py filename subreddit month_num year
# To continue filling an existing .py file, run with one argument:
#     collector.py filename

if len(sys.argv) == 5:
    # Create brand new .csv file.
    name = sys.argv[1]
    if not name.endswith('.csv'):
        name += '.csv'
    if path.exists(name):
        sys.exit("Whoops!! Already have a {}.".format(name))
    sub = sys.argv[2]
    month_num = int(sys.argv[3])
    year = int(sys.argv[4])
    next_date_pull = dt.datetime(year=year, month=month_num, day=1)
    with open(name, 'a', encoding='utf-8') as f:
        f.write(",".join(header) + "\n")
    batch_num = 0

elif len(sys.argv) == 2:
    # Continue with existing .csv file.
    name = sys.argv[1]
    if not name.endswith('.csv'):
        name += '.csv'
    if not path.exists(name):
        sys.exit("Whoops!! Don't have a {}.".format(name))
    temp = pd.read_csv(name,encoding="utf-8")
    sub = temp.iloc[0].subreddit
    print("Continuing to read from {}...".format(sub))
    next_date_pull = calculatePrevMonth(int(temp.date.min()-1))
    batch_num = temp.batch_num.max()

else:
    sys.exit(
        '''

        Welcome to collector.py.

        To create a new .py file, run with four arguments:
            collector.py filename subreddit month_num year
        To continue filling an existing .py file, run with one argument:
            collector.py filename
        ''')

BATCH_SIZE = 100

# Load known botnames.
bots = set(pd.read_csv("botnames.csv", squeeze=True, header=None))


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


subreddit = r.subreddit(sub)

#Log file - thread's TLC date and instantaneous system date (doesn't check for file existence)
log_header=['TLC_date','system_date']
l = open('collecting.log', 'a', encoding='utf-8')
l.write(",".join(log_header) + "\n")


with open(name, 'a', encoding='utf-8') as f:

    threads_this_month = 0

    while True:
        batch_num = batch_num + 1
        print("Starting new batch {} at time {}...".format(batch_num,
            makeDateReadable(next_date_pull,True)))
        posts = list(api.search_submissions(before=next_date_pull,
                    subreddit=sub, limit=BATCH_SIZE))

        #Loop through posts and put their threads in a csv file
        for post in posts:
            for top_level_comment in r.submission(post).comments:
                threads_this_month += 1
                if (hasattr(top_level_comment,"author") and
                    str(top_level_comment.author) in bots):
                    print("  (Discarding thread from known bot {})".format(
                        top_level_comment.author))
                elif hasattr(top_level_comment, "body"):
                    words = get_thread(top_level_comment, 0, "").replace('"', "'")
                    words = words.replace("\n", "\\n")
                    write = [sub, top_level_comment.link_id,
                    top_level_comment.id, '"' + words + '"',
                    str(top_level_comment.created_utc), str(batch_num)]
                    f.write(",".join(write) + "\n")
                    #writes to collecting.log TLC date and local time zone date
                    log_write = [str(top_level_comment.created_utc), str(datetime.now())]
                    l.write(",".join(log_write) + "\n")
                    f.flush()
                    l.flush()
                    print(" {}".format(
                        makeDateReadable(top_level_comment.created_utc,True)))
        if threads_this_month >= THREADS_PER_MONTH:
            next_date_pull = calculatePrevMonth(post.created_utc)
            threads_this_month = 0
        else:
            next_date_pull = int(post.created_utc-1)
    f.close()
    l.close()
