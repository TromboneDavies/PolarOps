

Final .csv looks like:

date,subreddit,comment_id,prediction,polar_score
3285728.0,ArizonaPolitics,h28782,polar,.897534
3285738.0,ArizonaPolitics,h28783,nonpolar,.19534


dates = np.array([])
subreddits = np.array([])
comment_ids = np.array([])
predictions = np.array([])
confidences = np.array([])


thefile = pd.read_csv(sys.argv[1])

for row in thefile.itertuples():
    do stuff with row.subreddit... row.comment_id...

    vec, _, _ = vectorize([row.text], None, vocab)
    trans_vec = selector.transform(vec).astype('float32')
    polar_score = model.predict(trans_vec)
    
    dates = np.append(dates, ......)
    subreddits = np.append(subreddits, ......)
    comment_ids = np.append(comment_ids, ......)
    polar_scores = np.append(polar_scores, polar_score)
    dates = np.append(dates, ......)
    


tjs_df = pd.DataFrame({ 'date':dates, 'subreddit':subreddits })
tjs_df.to_csv("")

