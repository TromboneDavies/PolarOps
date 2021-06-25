import praw
import datetime as dt
from psaw import PushshiftAPI

r = praw.Reddit("sensor1")
api = PushshiftAPI(r)


print(list(api.search_submissions(subreddit = 'politics', limit = 10)))
