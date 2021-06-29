import praw
import datetime as dt
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

#Questions for collection
sub = input("What is the name of the subreddit you are collecting (all lower case)?\n")

n = int(input("What is your limit?\n"))
date = input("What year would you like to pull from?")

start_epoch = int(dt.datetime(int(date), 1, 1).timestamp())
subreddit = r.subreddit(sub)

posts =  list(api.search_submissions(after=start_epoch, subreddit=sub,
                                                                limit=n))
#CSV file
with open('data.csv', 'a') as f:
    data = csv.writer(f)

    #Loop through posts and put their threads in a csv file
    final = ""
    comma = ","
    for post in posts:
        for top_level_comment in r.submission(post).comments:
            if hasattr(top_level_comment, "body"):
                words = get_thread(top_level_comment, 0, "").replace('"', "'")
                words = words.replace("\n", "\\n")
                write = [sub, top_level_comment.link_id, top_level_comment.id, '"' + words + '"', str(top_level_comment.created_utc)]
                f.write(comma.join(write) + "\n")

    f.flush()
    f.close()
