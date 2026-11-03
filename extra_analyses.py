"""Analyses behind Table V (AUC by distance-to-goal tercile) and Fig. 4 / label-B SHAP text.
Requires built.pkl, exp_raw.pkl (from feats.py, exp.py) and records2.pkl (fetch2.py). Writes err.json and shap2.json."""
import pickle, json, numpy as np, xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from feats import FEATS, GROUPS
d=pickle.load(open('exp_raw.pkl','rb')); S,P=pickle.load(open('built.pkl','rb')); recs=pickle.load(open('records2.pkl','rb'))
y=np.array([s[2] for s in S]); X=np.array([s[4] for s in S]); m=np.array([s[0] for s in S]); p=d['oof']['XGB']
dist=X[:,2]; q=np.quantile(dist,[1/3,2/3]); rng=np.random.default_rng(2); out=[]
for nm,sel in (('Near',dist<q[0]),('Mid',(dist>=q[0])&(dist<q[1])),('Far',dist>=q[1])):
    idx=np.where(sel)[0]; a=np.mean([roc_auc_score(y[idx],p[r][idx]) for r in range(3)])
    mi={u:idx[m[idx]==u] for u in np.unique(m[idx])}; ks=list(mi)
    bs=[np.mean([roc_auc_score(y[s_],p[r][s_]) for r in range(3)]) for s_ in (np.concatenate([mi[u] for u in rng.choice(ks,len(ks))]) for _ in range(200))]
    out.append((nm,len(idx),float(y[idx].mean()),float(a),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))))
json.dump(out,open('err.json','w')); print(out)
yB={r['id']:r['y_goal'] for r in recs if r['type']=='Shot'}; yb=np.array([yB[s[1]] for s in S]); res={}
for nm,yy in (('A',y),('B',yb)):
    mdl=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,eval_metric='logloss',random_state=42,n_jobs=1,tree_method='hist').fit(X,yy)
    c=mdl.get_booster().predict(xgb.DMatrix(X),pred_contribs=True)[:,:-1]; ma=np.abs(c).mean(0)
    res[nm]=dict(vals=ma.tolist(),groups={g:float(ma[i].sum()/ma.sum()) for g,i in GROUPS.items()},top=[(FEATS[i],float(ma[i])) for i in np.argsort(-ma)[:5]])
json.dump(res,open('shap2.json','w')); print({k:v['top'] for k,v in res.items()})
