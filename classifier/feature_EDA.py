#A lot of this is from TJ's classifier and I'm sure there's a better
#way to do this instead of repeating so much but if you run all
#I promise it works

import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def remove_punct(post):
    post = post.replace("\\n", "")
    post = post.replace(">>", "")
    post = post.replace(">", "")
    punctuation = string.punctuation 
    for element in punctuation:
        post = post.replace(element, "")
    post = post.replace("’", "")
    post = post.replace("—", "")
    post = post.replace("“", "")
    post = post.replace("”", "")
    return post.lower()

file_name = "training_data.csv"
posts = pd.read_csv(file_name, delimiter = ",")

# Initialize variables for future use
words = []
documents = []

# Iterate through dataframe's rows
for index in posts.index:
        # Remove punctuation/stopwords, tokenize, and stem
        legitWords = []
        post = remove_punct(posts['text'][index])
        for w in word_tokenize(post):
            if w not in stopwords.words('english'):
                legitWords.append(w)
                words.append(w)

        # Store edited string as a new labeled document
        documents.append((legitWords, posts['polarized'][index]))
        
        
pwords = [ x for (x,y) in documents if y == "yes" ]
npwords = [ x for (x,y) in documents if y == "no" ]

#----- Frequency of 20 most common words ----

from nltk.probability import FreqDist
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pwords = [ x for (x,y) in documents if y == "yes" ]
p = [ item for sublist in pwords for item in sublist ]
pfreq = FreqDist(p).most_common(20)
pfreq = pd.Series(dict(pfreq))
fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x=pfreq.index, y=pfreq.values, ax=ax).set(title='Frequency of the 20 Most Common Polarized Words')
plt.show()

npwords = [ x for (x,y) in documents if y == "no" ]
np = [ item for sublist in npwords for item in sublist ]
npfreq = FreqDist(np).most_common(20)
npfreq = pd.Series(dict(npfreq))
fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(x=npfreq.index, y=npfreq.values, ax=ax).set(title='Frequency of the 20 Most Common Non-Polarized Words')
plt.show()

#----- Average Word Length in a thread -----

from statistics import mean
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats

def wordlen(lis,empty):
    for x in lis:
        empty.append(mean([len(i) for i in x]))
        
p_wordlen=[]
np_wordlen=[]
wordlen(pwords,p_wordlen)
wordlen(npwords,np_wordlen)
p_wordlen=np.array(p_wordlen)
np_wordlen=np.array(np_wordlen)

print("The average Average Word Length of Polarized threads is {}".format(p_wordlen.mean()))
print("The average Average Word Length of Non-Polarized threads is {}".format(np_wordlen.mean()))

plt.hist(p_wordlen, range=(4,14), bins=15)
plt.title("Average Word Length of Polarized")
plt.show()
p_kde=scipy.stats.gaussian_kde(p_wordlen,bw_method=.3)
x_vals=np.arange(4,14,.1)
plt.plot(x_vals,p_kde(x_vals))
plt.title("Average Word Length of Polarized")
plt.show()

plt.hist(np_wordlen, range=(4,14), bins=15)
plt.title("Average Word Length of Non-Polarized")
plt.show()
np_kde=scipy.stats.gaussian_kde(np_wordlen,bw_method=.3)
plt.plot(x_vals,np_kde(x_vals))
plt.title("Average Word Length of Non-Polarized")
plt.show()

#----- Number of comments in a thread -----

def remove_punct_andadd(post):
    post = post.replace("\\n", "")
    post = post.replace(">>", " inthreadquote newcomment ")
    post = post.replace(">", " newcomment ")
    punctuation = string.punctuation 
    for element in punctuation:
        post = post.replace(element, "")
    post = post.replace("’", "")
    post = post.replace("—", "")
    post = post.replace("“", "")
    post = post.replace("”", "")
    return post.lower()

file_name = "training_data.csv"
posts = pd.read_csv(file_name, delimiter = ",")

# Initialize variables for future use
words = []
documents = []

# Iterate through dataframe's rows
for index in posts.index:
        # Remove punctuation/stopwords, tokenize, and stem
        legitWords = []
        post = remove_punct_andadd(posts['text'][index])
        for w in word_tokenize(post):
            if w not in stopwords.words('english'):
                legitWords.append(w)
                words.append(w)

        # Store edited string as a new labeled document
        documents.append((legitWords, posts['polarized'][index]))
        
pwords = [ x for (x,y) in documents if y == "yes" ]
npwords = [ x for (x,y) in documents if y == "no" ]

def num_com(lis,empty):
    for ele in lis:
        empty.append(ele.count('newcomment'))

p_num=[]
np_num=[]
num_com(pwords,p_num)
num_com(npwords,np_num)
p_num=np.array(p_num)
np_num=np.array(np_num)

print("The average Number of Comments in Polarized threads is {}".format(p_num.mean()))
print("The average Number of Comments in Non-Polarized threads is {}".format(np_num.mean()))

plt.hist(p_num, range=(0,50), bins=15)
plt.title("Number of Comments in Polarized")
plt.show()
p_kde=scipy.stats.gaussian_kde(p_num,bw_method=.3)
x_vals=np.arange(0,50,.1)
plt.plot(x_vals,p_kde(x_vals))
plt.title("Number of Comments in Polarized")
plt.show()

plt.hist(np_num, range=(0,50), bins=15)
plt.title("Number of Comments in Non-Polarized")
plt.show()
np_kde=scipy.stats.gaussian_kde(np_num,bw_method=.3)
plt.plot(x_vals,np_kde(x_vals))
plt.title("Number of Comments in Non-Polarized")
plt.show()
    
#----- Subreddits' balance between Polarized and Non-Polarized -----
a=pd.read_csv("training_data.csv")
b=pd.crosstab(a.polarized,a.Subreddit,margins=True)
c=(b.div(b.loc["All"],axis=1)*100)







