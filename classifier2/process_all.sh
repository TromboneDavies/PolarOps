#!/usr/bin/env bash

for file in /home/stephen/local/reddit/subreddit_chunk???.csv
do
    python3 process.py $file &
done
