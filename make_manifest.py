"""Writes data_manifest.csv (matches used, per competition, with shot/pass counts) and fold_assignments.csv
(match-level fold ids for repetitions 0-2, identical to the folds used in exp.py/exp2.py). Needs records2.pkl."""
import pickle, numpy as np, csv, collections
recs=pickle.load(open('records2.pkl','rb'))
cnt=collections.defaultdict(lambda:[None,0,0])
for r in recs:
    c=cnt[r['match']]; c[0]=r['comp']; c[1]+= r['type']=='Shot'; c[2]+= r['type']=='Pass'
matches=np.array(sorted(cnt)); K=5
folds=[]
for rep in range(3):
    perm=np.random.default_rng(rep).permutation(matches); fm={m:i%K for i,m in enumerate(perm)}; folds.append(fm)
with open('data_manifest.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['match_id','competition','n_shots','n_passes'])
    for m in matches: w.writerow([int(m),cnt[m][0],cnt[m][1],cnt[m][2]])
with open('fold_assignments.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['match_id','fold_rep0','fold_rep1','fold_rep2'])
    for m in matches: w.writerow([int(m)]+[folds[r][m] for r in range(3)])
print(len(matches),sum(v[1] for v in cnt.values()),sum(v[2] for v in cnt.values()))
