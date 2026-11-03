"""Reviewer-requested analyses, part A: (1) overlap between shot states and pass states, (2) calibration for shots and passes separately (goal label),
(3) feature-group ablation on the goal label. Needs records3.pkl (fetch3.py)."""
from common5 import *
from sklearn.linear_model import LogisticRegression
t0=time.time(); out={}
Y=np.array([q['yg'][10] for q in rows])             # primary label: goal within 10 s
# ---- (2) unified model OOF predictions, 3 repetitions
oof=np.zeros((R,len(Y)))
for rep in range(R):
    f=fold_of(rep)
    for k in range(K):
        tr=f!=k; te=f==k; oof[rep,te]=xgbm().fit(X[tr],Y[tr]).predict_proba(X[te])[:,1]
    print('unified rep',rep,round(time.time()-t0),flush=True)
def ece(y,p,nb=10):
    b=np.minimum((p*nb).astype(int),nb-1); return sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(nb) if (b==i).any())
def calib(y,p):
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
    lr=LogisticRegression(C=1e6,max_iter=1000).fit(z,y); return float(lr.coef_[0,0]),float(lr.intercept_[0])
cal={}
um=np.unique(M); mi={u:np.where(M==u)[0] for u in um}; rng=np.random.default_rng(3)
for nm,ii in (('shots',sh),('passes',ps)):
    inset=np.zeros(len(Y),bool); inset[ii]=True
    pt=[]; 
    for r in range(R):
        sl,ic=calib(Y[ii],oof[r][ii]); pt.append([brier_score_loss(Y[ii],oof[r][ii]),ece(Y[ii],oof[r][ii]),sl,ic,float(oof[r][ii].mean()),float(Y[ii].mean())])
    pt=np.mean(pt,axis=0); bs=[]
    for _ in range(200):
        s_=np.concatenate([mi[u] for u in rng.choice(um,len(um))]); s_=s_[inset[s_]]
        bs.append([ece(Y[s_],oof[0][s_]),calib(Y[s_],oof[0][s_])[0]])
    bs=np.array(bs)
    # reliability bins (rep 0, quantile bins)
    p=oof[0][ii]; y=Y[ii]; edges=np.unique(np.quantile(p,np.linspace(0,1,9))); idx=np.clip(np.digitize(p,edges[1:-1]),0,len(edges)-2)
    rel=[(float(p[idx==j].mean()),float(y[idx==j].mean()),int((idx==j).sum())) for j in range(len(edges)-1)]
    cal[nm]=dict(brier=pt[0],ece=pt[1],slope=pt[2],intercept=pt[3],mean_pred=pt[4],mean_obs=pt[5],ece_ci=[float(np.percentile(bs[:,0],2.5)),float(np.percentile(bs[:,0],97.5))],slope_ci=[float(np.percentile(bs[:,1],2.5)),float(np.percentile(bs[:,1],97.5))],reliability=rel)
    print(nm,{k:(round(v,4) if isinstance(v,float) else v) for k,v in cal[nm].items() if k!='reliability'},flush=True)
out['calibration']=cal
# ---- (1) overlap: domain classifier on carrier-state features only (14), shot vs pass
Xd=X[:,:14]; yd=(T==0).astype(int); pd_=np.zeros((R,len(yd)))
for rep in range(R):
    f=fold_of(rep)
    for k in range(K):
        tr=f!=k; te=f==k
        mdl=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.75,scale_pos_weight=float((yd==0).sum()/(yd==1).sum()),eval_metric='logloss',random_state=42,n_jobs=1,tree_method='hist').fit(Xd[tr],yd[tr])
        pd_[rep,te]=mdl.predict_proba(Xd[te])[:,1]
pdm=pd_.mean(0)            # balanced-prior probability that a state is a shot state
out['overlap']=dict(domain_auc=float(np.mean([roc_auc_score(yd,pd_[r]) for r in range(R)])),
   shot_p_quantiles={q:float(np.quantile(pdm[sh],q)) for q in (0.05,0.25,0.5,0.75,0.95)},pass_p_quantiles={q:float(np.quantile(pdm[ps],q)) for q in (0.05,0.25,0.5,0.75,0.95)},
   share_shots_p_gt={str(c):float((pdm[sh]>c).mean()) for c in (0.8,0.9,0.95)},share_passes_p_gt={str(c):float((pdm[ps]>c).mean()) for c in (0.8,0.9,0.95)})
print(out['overlap'],flush=True)
# ---- CDS per shot (rep 0, primary label), with overlap trimming
f=fold_of(0); SH=[i for i in sh if len(rows[i]['tm'])>0]
cds={}; psh={}; ppass={}
for k in range(K):
    tr=f!=k; te=f==k; mdl=xgbm().fit(X[tr],Y[tr]); ev=[i for i in SH if te[i]]
    C_=[cand_basic(rows[i]['loc'],rows[i]['tm'],rows[i]['op'],rows[i]['gk']) for i in ev]
    pp=mdl.predict_proba(np.vstack([c[0] for c in C_]))[:,1]; ps_=mdl.predict_proba(X[ev])[:,1]; off=0
    for z,i in enumerate(ev):
        nk=len(C_[z][0]); prk=pp[off:off+nk]; off+=nk; j=int(np.argmax(C_[z][1]@W0)); cds[i]=prk[j]-ps_[z]; psh[i]=ps_[z]; ppass[i]=prk[j]
v=np.array([cds[i] for i in SH]); p_overlap=pdm[SH]
def summ(m): return dict(n=int(m.sum()),mean_cds=float(v[m].mean()),median_cds=float(np.median(v[m])),pos=float((v[m]>0).mean()),mean_pshot=float(np.mean([psh[SH[z]] for z in np.where(m)[0]])),mean_ppass=float(np.mean([ppass[SH[z]] for z in np.where(m)[0]])))
out['cds_overlap']={'all':summ(np.ones(len(SH),bool)),'overlap p<=0.95':summ(p_overlap<=0.95),'overlap p<=0.90':summ(p_overlap<=0.90),'overlap p<=0.80':summ(p_overlap<=0.80),'poor overlap p>0.95':summ(p_overlap>0.95) if (p_overlap>0.95).any() else None}
print(json.dumps(out['cds_overlap']),flush=True)
np.savez('cds_rep0.npz',idx=np.array(SH),cds=v,pshot=np.array([psh[i] for i in SH]),ppass=np.array([ppass[i] for i in SH]),p_shot_state=p_overlap)
# ---- (3) ablation on the goal label (shot-only model, paired bootstrap over matches)
Xs=X[sh][:,:14]; Ys=Y[sh]; Ms=M[sh]; ums=np.unique(Ms); mis={u:np.where(Ms==u)[0] for u in ums}
full=np.zeros((R,len(sh))); abl={g:np.zeros((R,len(sh))) for g in GROUPS}
for rep in range(R):
    fs=fold_of(rep)[sh]
    for k in range(K):
        tr=fs!=k; te=fs==k; full[rep,te]=xgbm().fit(Xs[tr],Ys[tr]).predict_proba(Xs[te])[:,1]
        for g,idxs in GROUPS.items():
            keep=[i for i in range(14) if i not in idxs]; abl[g][rep,te]=xgbm().fit(Xs[tr][:,keep],Ys[tr]).predict_proba(Xs[te][:,keep])[:,1]
rng=np.random.default_rng(21); draws=[np.concatenate([mis[u] for u in rng.choice(ums,len(ums))]) for _ in range(300)]
ab={'full_auc':float(np.mean([roc_auc_score(Ys,full[r]) for r in range(R)]))}
for g in GROUPS:
    pt=np.mean([roc_auc_score(Ys,abl[g][r])-roc_auc_score(Ys,full[r]) for r in range(R)])
    bs=[np.mean([roc_auc_score(Ys[d],abl[g][r][d])-roc_auc_score(Ys[d],full[r][d]) for r in range(R)]) for d in draws]
    ab[g]=[float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out['ablation_goal']=ab; print(ab,flush=True)
json.dump(out,open('results5a.json','w'),indent=1); print('done',round(time.time()-t0))
