
# To run this, you must:
# - pip install praw
# - create an account on reddit.com
# - on https://www.reddit.com/prefs/apps, click "create another app." It should
#    be a "script" app, and you can name it anything you want.
# - your "client ID" (see below) is the "personal use script" string, and your
#    "client secret" is the "secret" string. Copy these.

# Make a praw.ini file (in your config folder; on Linux, ~/.config) and make 
# it look like this:
#
#[sensor1]
#client_id=yourclientid
#client_secret=yourclientsecret
#user_agent=justmakesomethingupreally
#
# 
# Or, to hardcode directly in this file instead (not recommended):
#reddit = praw.Reddit(
#    client_id="the client ID",
#    client_secret="the client Secret",
#    user_agent="justmakesomethingupreally"
#)
#
# (See https://praw.readthedocs.io/en/latest/getting_started/quick_start.html)


import praw


def print_submissions(subs):
    for sub in subs:
        print("\n---------------------------------------------------------")
        print(sub.title + "\n")
        if "selftext" in vars(sub):
            print(sub.selftext)
        else:
            print("No text.")


# Create a "Reddit" object, using a particular bot specified in praw.ini.
reddit = praw.Reddit("sensor1")

# Create a "Subreddit" object for the r/textdatamining subreddit.
tdm = reddit.subreddit("textdatamining")

#print("=== new r/textdatamining submissions ===")
#print_submissions(tdm.new(limit=10))

print("=== r/textdatamining submissions matching 'nltk' ===")
#print_submissions(tdm.search("selftext:nltk",limit=10))

print("=== ALL submissions matching a search string ===")
print_submissions(reddit.subreddit("all").search("selftext:sheri tepper",limit=10))
