
# Proof-of-concept code to suck all replies-to-replies-to-...-to-replies of a
# submissions' comment. (This aggregate text we call a "thread.")

# (Reminder: a "submission" has a "title," but a "comment" only has a "body.")

import praw

# Authenticate to Reddit.
reddit = praw.Reddit("sensor1")

# Get a Subreddit.
sw = reddit.subreddit("starwars")

# Get the 30 hottest submissions as of now.
hot = [ h for h in sw.hot(limit=30) ]

# If you wish, display each of them with their index value.
#print([(i,h.title) for i,h in enumerate(hot) ])

# Let's dig in to hot submission #10. Here is its title:
print("'Hot' submission #10 is: {}\n".format(hot[10].title))

# Here's how many (direct) comments it has:
print("It has {} comments.\n".format(len(hot[10].comments)))

# We'll look at the first comment:
first_comm = [c for c in hot[10].comments][0]
print("The first comment says: {}\n".format(first_comm.body))

# Here's how many replies there are to that first comment of the 10th hottest
# submission:
print("That comment has {} replies.\n".format(len(first_comm._replies)))

# And dig down into its first reply:
first_reply = [ x for x in first_comm._replies ][0]
print("Its 1st reply is: {}\n".format(first_reply.body))

# Here's how many replies there are to THAT reply:
print("That reply has {} replies.\n".format(len(first_reply._replies)))

# And dig down into ITS first reply:
first_reply_to_first_reply = [ x for x in first_reply._replies ][0]
print("Its 1st reply is: {}\n".format(first_reply_to_first_reply.body))




# Output as of 6/7/2021:
#
#'Hot' submission #10 is: Just finished watching Star Wars: The Clone Wars. How do I deal with the existential crisis that I now have?
#It has 159 comments.
#The first comment says: Start watching The Bad Batch
#That comment has 14 replies.
#Its 1st reply is: Yeah that's what I'm planning on doing. Is the show made in the same style as The Clone Wars or is it different?
#That reply has 13 replies.
#Its 1st reply is: Same style just another jump in animation quality since Season 7
