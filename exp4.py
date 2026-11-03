import numpy as np, pickle, json, time
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from feats import feat14, parse, GC, seg_dist, FEATS, GROUPS
t0=time.time(); R,K,B=3,5,300
recs=pickle.load(open('records3.pkl','rb'))
def xgbm(seed=42): return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,eval_metric='logloss',random_state=seed,n_jobs=1,tree_method='hist')
def pfe(loc,end,opall):
    d=np.linalg.norm(end-loc); lane=seg_dist(opall,loc,end).min() if len(opall) else np.nan; free=np.linalg.norm(opall-end,axis=1).min() if len(opall) else np.nan
    return np.array([d,lane,free,np.linalg.norm(end-GC),np.linalg.norm(loc-GC)-np.linalg.norm(end-GC)])
def opall_of(op,gk): return op if gk is None else (np.vstack([op,gk[None]]) if len(op) else gk[None])
SX,SY=105/120,68/80
rows=[]; 
for r in recs:
    loc,tm,op,gk=parse(r); oa=opall_of(op,gk); base=feat14(loc,tm,op,gk)
    if r['type']=='Shot':
        x=np.concatenate([base,[0],np.full(5,np.nan)]); rows.append(dict(m=r['match'],c=r['comp'],t=0,x=x,yA=r['y'],yB=r['y_goal'],yg=r['yg'],own=(r.get('sh_out')=='Goal'),xg=r['xg'],loc=loc,tm=tm,op=op,gk=gk,id=r['id']))
    else:
        end=np.array([r['end'][0]*SX,r['end'][1]*SY]); x=np.concatenate([base,[1],pfe(loc,end,oa)])
        rows.append(dict(m=r['match'],c=r['comp'],t=1,x=x,yA=r['y'],yB=r['y_goal'],yg=r['yg'],own=(r.get('sh_out')=='Goal'),ok=r['pass_ok'],end=end,loc=loc,tm=tm,op=op,gk=gk,id=r['id']))
X=np.array([q['x'] for q in rows]); yA=np.array([q['yA'] for q in rows]); yB=np.array([q['yB'] for q in rows]); T=np.array([q['t'] for q in rows]); M=np.array([q['m'] for q in rows]); C=np.array([q['c'] for q in rows])
sh=np.where(T==0)[0]; ps=np.where(T==1)[0]; XG=np.array([rows[i]['xg'] for i in sh])
out={'n_shots':len(sh),'n_pass':len(ps),'rate':{'A_shot':float(yA[sh].mean()),'A_pass':float(yA[ps].mean()),'B_shot':float(yB[sh].mean()),'B_pass':float(yB[ps].mean())}}

matches=np.array(sorted(set(M)))
def fold_of(rep):
    perm=np.random.default_rng(rep).permutation(matches); fm={m:i%K for i,m in enumerate(perm)}; return np.array([fm[m] for m in M])
def m4(y,p):
    b=np.minimum((p*10).astype(int),9); e=sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(10) if (b==i).any())
    return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),e])
def pairs(a,b):  # candidate features for counterfactual passes
    pass

import json
from scipy.stats import spearmanr
W0=np.array([.25]*4)
def cand(loc,tm,op,gk):
    oa=opall_of(op,gk); base=feat14(loc,tm,op,gk); F=[];comp=[]
    for j in range(len(tm)):
        end=tm[j]; pf=pfe(loc,end,oa); F.append(np.concatenate([base,[1],pf]))
        comp.append([1-min(pf[0]/30,1),min((10 if np.isnan(pf[1]) else pf[1])/10,1),min((10 if np.isnan(pf[2]) else pf[2])/10,1),np.clip(pf[4]/20,0,1)])
    return np.array(F),np.array(comp)
# candidate features are label independent: compute once for all shots with a visible teammate
SH=[i for i in sh if len(rows[i]['tm'])>0]; CF={i:cand(rows[i]['loc'],rows[i]['tm'],rows[i]['op'],rows[i]['gk']) for i in SH}
labels={f'goal<={h}s':np.array([q['yg'][h] for q in rows]) for h in (5,10,15,20,30)}
yD=np.array([ (0 if q['own'] else q['yg'][10]) if q['t']==0 else q['yg'][10] for q in rows]); labels['downstream-only (excl. own goal), 10s']=yD
f=fold_of(0); res={}
for nm,Y in labels.items():
    oof=np.zeros(len(Y)); cds={}
    for k in range(K):
        tr=f!=k; te=f==k; mdl=xgbm().fit(X[tr],Y[tr]); oof[te]=mdl.predict_proba(X[te])[:,1]
        ev=[i for i in SH if te[i]]
        if not ev: continue
        pp=mdl.predict_proba(np.vstack([CF[i][0] for i in ev]))[:,1]; psh=mdl.predict_proba(X[ev])[:,1]; off=0
        for z,i in enumerate(ev):
            nk=len(CF[i][0]); prk=pp[off:off+nk]; off+=nk; cds[i]=(prk[np.argmax(CF[i][1]@W0)]-psh[z],psh[z],prk[np.argmax(CF[i][1]@W0)])
    v=np.array([cds[i][0] for i in SH]); xs=np.array([rows[i]['xg'] for i in SH])
    res[nm]=dict(shot_rate=float(Y[sh].mean()),pass_rate=float(Y[ps].mean()),auc_shot=float(roc_auc_score(Y[sh],oof[sh])),auc_pass=float(roc_auc_score(Y[ps],oof[ps])),
        mean_pshot=float(np.mean([cds[i][1] for i in SH])),mean_ppass=float(np.mean([cds[i][2] for i in SH])),mean_cds=float(v.mean()),median_cds=float(np.median(v)),pos=float((v>0).mean()),rho_xg=float(spearmanr(v,xs)[0]))
    print(nm,{k:round(x,3) for k,x in res[nm].items()},flush=True)
res['own_goal_share_of_shot_label_B']=float(np.mean([rows[i]['own'] for i in sh])/np.mean([rows[i]['yg'][10] for i in sh]))
json.dump(res,open('results4.json','w'),indent=1); print('done',round(time.time()-t0))
