import praw
import datetime as dt
from psaw import PushshiftAPI

r = praw.Reddit("sensor1")
api = PushshiftAPI(r)

start_epoch=int(dt.datetime(2010, 1, 1).timestamp())

posts =  list(api.search_submissions(before=start_epoch, subreddit='politics',
                                                                limit=1000000))
f = open("data.txt", "w")

for post in posts:
    for top_level_comment in r.submission(post).comments:
        if hasattr(top_level_comment, "body"):
            if len(top_level_comment.body.split(" ")) > 7:

                f.write(top_level_comment.body)
                f.write("\n-----\n")

f.close()
