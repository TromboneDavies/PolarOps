
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
import os
import sys
import subprocess

# The file in which to put reconciled training examples. This script will
# append to the file if it already exists.
TRAINING_FILE = "training_data.csv"

if len(sys.argv) != 2:
    print("Usage: analyze.py completedPiggiedFile.csv.\n")
    sys.exit(1)

if not os.path.isfile(sys.argv[1]):
    print("No such file {}.\n".format(sys.argv[1]))
    sys.exit(2)

if not sys.argv[1].startswith("piggied"):
    print("ERROR: This script should only be run on a piggied file!\n")
    sys.exit(3)

data = pd.read_csv(sys.argv[1])


# Clean up clutter we don't need for now.
#del data['date']
#del data['SubmissionID']

datal = pd.melt(data, ['Subreddit','SubmissionID','CommentID','text','date'],
    var_name='Assessor', value_name='ttype')
datal['community'] = np.where(datal.ttype.isin([1,2]),"hetero","homo")
datal['polarized'] = np.where(datal.ttype.isin([1,4]),"yes","no")

if any(datal.ttype == -99):
    print("Incomplete file! Missing ratings from {}.".format(
        ",".join(datal[datal.ttype == -99].Assessor.unique())))
    print("")
    sys.exit(4)

write_header = False
if os.path.isfile(TRAINING_FILE):
    x = subprocess.run("wc -l {}".format(TRAINING_FILE).split(),
        capture_output=True)
    print("Continuing to append to {} (currently {} lines).".
        format(TRAINING_FILE, x.stdout.decode('utf-8').split()[0]))
    input("Press Enter to continue.")
else:
    print("Creating new file {}.".format(TRAINING_FILE))
    input("Press Enter to continue.")
    write_header = True

with open(TRAINING_FILE,"a",encoding="utf-8") as f:
    if write_header:
        print("Subreddit,SubmissionID,CommentID,text,date,ttype,community,"+
            "polarized",file=f)
        f.flush()
    num_threads = len(datal.CommentID.unique())
    for i,commentID in enumerate(datal.CommentID.unique()):
        print("\n=========================================================\n")
        first_row = datal[datal.CommentID == commentID].iloc[0]
        print("Comment ID: {}, from Subreddit {}\n".format(commentID,
            first_row.Subreddit))
        print(first_row.text.replace("\\n","\n"))
        result = datal[datal.CommentID == commentID][['Assessor','ttype',
            'community', 'polarized']].set_index('Assessor')
        print("Thread {} of {}: ".format(i+1,num_threads),end="")
        if len(result.drop_duplicates()) == 1:
            print("\nUNANIMOUS!")
            print(result)
            ttype = str(first_row.ttype)
            input("Press Enter to accept.")
        else:
            print("\nDiscrepancy:")
            print(result)
            ttype = input("Choose 1/2/3/4 ttype, " +
                "or 0 to exclude from training data. ")
        if ttype != "0":
            community = "hetero" if int(ttype) in [1,2] else "homo"
            polarized = "yes" if int(ttype) in [1,4] else "no"
            print(",".join([
                first_row.Subreddit,
                first_row.SubmissionID,
                first_row.CommentID,
                "\"" + first_row.text + "\"",
                str(first_row.date),
                ttype,
                community,
                polarized]),file=f)
            f.flush()
f.close()
