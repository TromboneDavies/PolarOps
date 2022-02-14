import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
import sys
from os.path import exists

# The number of "positive" (non-"other") votes a unanimous thread must receive
# from the 8 raters in order to count as usable.
MAX_VOTES_REQ = 4

pd.set_option('display.width',180)
pd.set_option('max.columns',10)

m = pd.read_csv("minratings.csv")
#m = m[m.rater != 'stephen']
m = m.drop_duplicates(['comment_id','rater'])

ct = pd.crosstab(m.rater,m.rating,margins=True)
ct1 = 100*(ct.div(ct['All'],axis=0)).round(2)
ct2 = 100*(ct.div(ct.loc['All'],axis=1)).round(2)

btraw = pd.crosstab(m.comment_id,m.rating)
bt = btraw[["polarized","notpolarized"]]
bt = bt.value_counts().sort_index()

# "u"sable, "r"ejected
uns = np.array([],dtype=int)
ups = np.array([],dtype=int)
ucs = np.array([],dtype=int)
rns = np.array([],dtype=int)
rps = np.array([],dtype=int)
rcs = np.array([],dtype=int)

for p,n in [(p,n) for p in range(10) for n in range(8) if (p==0) != (n==0)]:
    if (p,n) in bt:
        uns = np.append(uns,n)
        ups = np.append(ups,p)
        ucs = np.append(ucs,bt[(p,n)])

for p,n in [(p,n) for p in range(10) for n in range(8) if p>0 and n>0]:
    if (p,n) in bt:
        rns = np.append(rns,n)
        rps = np.append(rps,p)
        rcs = np.append(rcs,bt[(p,n)])

temp = pd.DataFrame({'notpolarized':uns,'polarized':ups,'total':ucs})
usable = temp[(temp.polarized>=MAX_VOTES_REQ)|(temp.notpolarized>=MAX_VOTES_REQ)]
later = temp[~temp.index.isin(usable.index)]
rejected = pd.DataFrame({'notpolarized':rns,'polarized':rps,'total':rcs})

print(m.rater.value_counts())
if input("...") == 'q': sys.exit()
tendencies_raw = pd.crosstab(m.rater,m.rating,margins=True)
tendencies = ((100 * tendencies_raw.div(tendencies_raw['All'],axis=0)).round(1)
    )[['polarized','notpolarized','other']]
vc = m.rater.value_counts()
vc['All'] = vc.sum()
tendencies['num'] = vc
print(tendencies)
if input("...") == 'q': sys.exit()
num_usable = sum(usable.total)
num_later = sum(later.total)
num_rejected = sum(rejected.total)

print(f"Immediately rejecting {sum(rejected.total)} non-unanimous threads.")
print(f"Sadly discarding {num_later} threads that are unanimous but have fewer than {MAX_VOTES_REQ} votes for polar/non.")
print(f"Can use {num_usable} threads!")
if exists("fall.csv"):
    print("fall.csv already exists! Not overwriting.")
else:
    print(f"Writing those to fall.csv...")
    fall = btraw[((btraw.notpolarized==0)&(btraw.polarized>=MAX_VOTES_REQ))|
          ((btraw.polarized==0)&(btraw.notpolarized>=MAX_VOTES_REQ))]
    (fall.polarized>0).to_csv("fall.csv")
    
if input("...") == 'q': sys.exit()


def compute_irr(tdf, raters):
    filtered = tdf[tdf.rater.isin(raters)]
    ir_data = filtered.pivot(index='comment_id',columns='rater',values='rating')
    ir_data_complete = ir_data.dropna()
    agg = aggregate_raters(ir_data_complete)
    return len(ir_data_complete), fleiss_kappa(agg[0])

num, kappa = compute_irr(m, m.rater.unique())
print(f"Our overall Fleiss κ (on {num} threads): {kappa}")
if input("...") == 'q': sys.exit()

# (28,8) is the shape of ir_data_complete when all raters must be included
sanity = np.random.choice(['polarized','notpolarized','other'],(28,8))
sanity_agg = aggregate_raters(sanity)
print(f"Random Fleiss κ: {fleiss_kappa(sanity_agg[0])}")
if input("...") == 'q': sys.exit()

raters = m.rater.unique()
num_mat = np.empty((len(raters),len(raters)))
num_mat[:] = np.nan
irr_mat = np.ones((len(raters),len(raters)))
for r1 in range(len((raters))):
    for r2 in range(len((raters))):
        if r1 != r2:
            num_mat[r1][r2], irr_mat[r1][r2] = compute_irr(m, [raters[r1],raters[r2]])

num_df = pd.DataFrame(num_mat, index=raters, columns=raters)
irr_df = pd.DataFrame(irr_mat, index=raters, columns=raters).round(2)
print(irr_df)
plt.clf()
fig,ax = plt.subplots()
sns.heatmap(num_df, annot=True, vmin=0, fmt="g")
plt.title("Number of ratings for each pair of raters")
fig.tight_layout()
plt.savefig("num_ratings.png")
plt.clf()
fig,ax = plt.subplots()
sns.heatmap(irr_df, annot=True, vmin=0, vmax=1)
plt.title("Fleiss κ for each pair of raters")
fig.tight_layout()
plt.savefig("irr_ratings.png")
    
#raters = np.array([],dtype=object)
#nump = np.array([],dtype=float)
#numnp = np.array([],dtype=float)
#numo = np.array([],dtype=float)
#for rater in m.rater.unique():
#    np.append(raters, rater)
#    np.append(nump, len(m[m.rater==rater,m.rating='polarized']))
#    np.append(numnp, len(m[m.rater==rater,m.rating='notpolarized']))
#    np.append(numo, len(m[m.rater==rater,m.rating='other']))

