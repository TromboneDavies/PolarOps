import praw

def get_thread(top_level_comment, tab, final):
    if hasattr(top_level_comment, "body"):
        for i in range(tab):
            final = final + "\t"

        final = final + ">" + top_level_comment.body + "\n"

        for comment in top_level_comment._replies:
            final = get_thread(comment, tab + 1, final)

    return final

reddit = praw.Reddit("sensor1")

sub = input("What is the name of the reddit you would like to handtag?\n")
tdm = reddit.subreddit(sub)
name = input("Where would you like your data to go? (Data will be appended)\n")
f = open(name, "a", encoding = 'utf-8')
user = input("\nType 1/2/3/4 after each thread to designate which type it is.\n\nYou can type \"exit\" to stop the program.\n\nPlease type your name to begin.\n") 

tt = ""
legit = ["1", "2", "3", "4"]
comma = ","

for submission in tdm.hot(limit = 500):
    submission.comments.replace_more(limit=0)
    for top_level_comment in reddit.submission(submission).comments:
        print("------------------------------------------------------------")
        words = get_thread(top_level_comment, 0, "").replace('"', "'")
        print(words)
        words = words.replace("\n", " ")
        words = words.replace("\t", "")
        tt = input()

        if tt == "exit":
            break

        if tt in legit:
            write = [sub, top_level_comment.link_id, top_level_comment.id, '"' + words + '"', str(tt), user, str(top_level_comment.created_utc)]
            f.write(comma.join(write) + "\n")

    if tt == "exit":
        break
