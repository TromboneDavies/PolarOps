
import requests
import json
import urllib

api_key = "see discord"

comments_base_url = "https://api.nytimes.com/svc/community/v3/user-content/url.json?api-key={}&offset=0&url=".format(api_key)

articles_base_url = "https://api.nytimes.com/svc/search/v2/articlesearch.json?api-key={}".format(api_key)

articles_url = "{}&q={}".format(
    articles_base_url,
    "kamala")

res = json.loads(requests.get(articles_url).content.decode('utf-8'))
urls = [ j['web_url'] for j in res['response']['docs']
        if '/video/' not in j['web_url'] ]
urls = urls

commentss = []
repliess = []

for url in urls:
    comments_url = comments_base_url + urllib.parse.quote_plus(url)
    comments = json.loads(requests.get(comments_url).content.decode('utf-8'))
    if comments['errorDetails'] == '':
        for comment in comments['results']['comments']:
            commentss.append(comment['commentBody'])
            for reply in comment['replies']:
                repliess.append(reply['commentBody'])
    
with open("nyt_comments.txt","w") as f:
    json.dump(commentss,fp=f)
with open("nyt_replies.txt","w") as f:
    json.dump(repliess,fp=f)

