import praw
import datetime as dt
from psaw import PushshiftAPI
import csv

r = praw.Reddit("sensor1")
api = PushshiftAPI(r)

start_epoch=int(dt.datetime(2010, 1, 1).timestamp())

#Questions for collection
sub = input("What is the name of the subreddit you are collecting (all lower case)?\n")
n = input("What is your limit?\n")
subreddit = r.subreddit(sub)
posts =  list(api.search_submissions(after=start_epoch, subreddit='sub',
                                                                limit=n))

    
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

#CSV file
with open('data.csv', 'w') as f:
    data = csv.writer(f)

#Loop from TJ OG collector.py, but I kinda tried
for post in posts:
    for top_level_comment in r.submission(post).comments:
        if hasattr(top_level_comment, "body"):
            if len(top_level_comment.body.split(" ")) > 7:
#               ***
                f.writerow(top_level_comment.body)

f.close()




# *** okay so in my mind, somehow everything below would go where the stars are above
#did i necessarily think it was going to work, no, but I thought maybe it would do something,
#but I am now realizing that everything below pulls from most recent so it can't 
#just be hodge podged together

f = open("data.csv", "w", encoding = 'utf-8')

comma = ","

#For every top comment in a submission, take its thread
for submission in subreddit.hot(limit = n):
    submission.comments.replace_more(limit=0)
    for top_level_comment in r.submission(submission).comments:
        words = get_thread(top_level_comment, 0, "").replace('"', "'")
        words = words.replace("\n", "\\n")
        #subreddit, submission id, comment id, text, date
        write = [sub, top_level_comment.link_id, top_level_comment.id, '"' + words + '"', str(top_level_comment.created_utc)]
        f.write(comma.join(write) + "\n")


f.flush()
f.close()
