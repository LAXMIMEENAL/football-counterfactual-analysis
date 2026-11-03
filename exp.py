import numpy as np, pickle, time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from feats import *
t0=time.time()
S,P=pickle.load(open('built.pkl','rb'))
Xs=np.array([s[4] for s in S]); ys=np.array([s[2] for s in S]); ms=np.array([s[0] for s in S])
Xp=np.array([p[3] for p in P]); yp=np.array([p[2] for p in P]); mp=np.array([p[0] for p in P])
matches=np.array(sorted(set(ms)|set(mp)))
R,K=3,5; NJ=1
def xgb(kind='ext',seed=42):
    if kind=='pilot': return XGBClassifier(n_estimators=150,max_depth=2,learning_rate=0.03,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,scale_pos_weight=0.7,eval_metric='logloss',random_state=seed,n_jobs=NJ,tree_method='hist')
    return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,scale_pos_weight=1.0,eval_metric='logloss',random_state=seed,n_jobs=NJ,tree_method='hist')
def lr(): return make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(max_iter=2000))
def rf(): return make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=200,min_samples_leaf=5,n_jobs=NJ,random_state=42))
def folds(rep):
    rng=np.random.default_rng(rep); perm=rng.permutation(matches); return {m:i%K for i,m in enumerate(perm)}
MODELS={'LR':lr,'RF':rf,'XGB-pilot':lambda:xgb('pilot'),'XGB':xgb}
oof={k:np.zeros((R,len(ys))) for k in MODELS}
abl={g:np.zeros((R,len(ys))) for g in GROUPS}
oofp={'XGB-17':np.zeros((R,len(yp))),'XGB-14':np.zeros((R,len(yp)))}
thr_res=[]; cds_store=[]
W0=np.array([.25,.25,.25,.25]); pert=[]
for i in range(4):
    for sgn in (1,-1):
        w=W0.copy(); w[i]+=sgn*0.1; pert.append(w/w.sum())
def cand_feats(loc,tm,op,gk):
    opall=op if gk is None else (np.vstack([op,gk[None]]) if len(op) else gk[None])
    cand=[];comp=[]
    for j in range(len(tm)):
        end=tm[j]; others=np.vstack([np.delete(tm,j,0),loc[None]])
        f=np.concatenate([feat14(end,others,op,gk),pass3(loc,end,opall)]); cand.append(f)
        dk=min(np.linalg.norm(end-loc)/30,1); ck=min((10 if np.isnan(f[15]) else f[15])/10,1); rk=min((10 if np.isnan(f[16]) else f[16])/10,1)
        pk=np.clip((np.linalg.norm(loc-GC)-np.linalg.norm(end-GC))/20,0,1); comp.append([1-dk,ck,rk,pk])
    return np.array(cand),np.array(comp)
for rep in range(R):
    fm=folds(rep); fs=np.array([fm[m] for m in ms]); fp=np.array([fm[m] for m in mp])
    N=len(ys); keys=['pshot','top1','top3min','top3max','near','wstab','nstab','ppass1','pnear']
    cd={k:np.full(N,np.nan) for k in keys}
    for k in range(K):
        tr=fs!=k; te=fs==k; trp=fp!=k; tep=fp==k
        for name,mk in MODELS.items():
            m=mk(); m.fit(Xs[tr],ys[tr]); oof[name][rep,te]=m.predict_proba(Xs[te])[:,1]
        for g,idx in GROUPS.items():
            keep=[i for i in range(14) if i not in idx]
            m=xgb(); m.fit(Xs[tr][:,keep],ys[tr]); abl[g][rep,te]=m.predict_proba(Xs[te][:,keep])[:,1]
        mp17=xgb(); mp17.fit(Xp[trp],yp[trp]); oofp['XGB-17'][rep,tep]=mp17.predict_proba(Xp[tep])[:,1]
        mp14=xgb(); mp14.fit(Xp[trp][:,:14],yp[trp]); oofp['XGB-14'][rep,tep]=mp14.predict_proba(Xp[tep][:,:14])[:,1]
        trm=sorted(set(ms[tr])); inner={m:i%3 for i,m in enumerate(np.random.default_rng(100+rep*10+k).permutation(trm))}
        fi=np.array([inner[m] for m in ms[tr]]); Xt,yt=Xs[tr],ys[tr]; po=np.zeros(len(yt))
        for j in range(3):
            mm=xgb(); mm.fit(Xt[fi!=j],yt[fi!=j]); po[fi==j]=mm.predict_proba(Xt[fi==j])[:,1]
        ths=np.linspace(0.05,0.8,76); th=ths[np.argmax([balanced_accuracy_score(yt,po>=t) for t in ths])]
        pr=oof['XGB'][rep,te]; yy=ys[te]; b=pr>=th
        thr_res.append([th,accuracy_score(yy,b),balanced_accuracy_score(yy,b),precision_score(yy,b,zero_division=0),recall_score(yy,b),f1_score(yy,b)])
        smod=xgb(); smod.fit(Xs[tr],ys[tr])
        evs=[e for e in np.where(te)[0] if len(S[e][6])>0]
        rng=np.random.default_rng(1000+rep*10+k)
        feats_c=[];recs=[];noisy=[]
        for ei in evs:
            _,_,_,_,fa,loc,tm,op,gk,_c=S[ei]
            cand,comp=cand_feats(loc,tm,op,gk); recs.append((ei,len(cand),comp,np.linalg.norm(tm-loc,axis=1))); feats_c.append(cand)
            nl=loc+rng.normal(0,1,2); ntm=tm+rng.normal(0,1,tm.shape); nop=op+rng.normal(0,1,op.shape) if len(op) else op; ngk=None if gk is None else gk+rng.normal(0,1,2)
            nc,ncomp=cand_feats(nl,ntm,nop,ngk); noisy.append((feat14(nl,ntm,nop,ngk),nc,ncomp))
        pp=mp17.predict_proba(np.vstack(feats_c))[:,1]
        psh=smod.predict_proba(Xs[evs])[:,1]
        nsh=smod.predict_proba(np.array([e[0] for e in noisy]))[:,1]
        nall=mp17.predict_proba(np.vstack([e[1] for e in noisy]))[:,1]
        off=0;noff=0
        for n,(ei,nk,comp,dists) in enumerate(recs):
            prk=pp[off:off+nk]; off+=nk; ps=psh[n]; cds=prk-ps; order=np.argsort(-(comp@W0))
            cd['pshot'][ei]=ps; cd['top1'][ei]=cds[order[0]]; cd['ppass1'][ei]=prk[order[0]]
            t3=cds[order[:3]]; cd['top3min'][ei]=t3.min(); cd['top3max'][ei]=t3.max()
            nn=np.argmin(dists); cd['near'][ei]=cds[nn]; cd['pnear'][ei]=prk[nn]
            base=np.sign(cds[order[0]])
            cd['wstab'][ei]=float(all(np.sign(cds[np.argmax(comp@w)])==base for w in pert))
            nsc=noisy[n][2]@W0; npk=nall[noff:noff+len(nsc)]; noff+=len(nsc)
            cd['nstab'][ei]=float(np.sign(npk[np.argmax(nsc)]-nsh[n])==base)
        print('rep',rep,'fold',k,round(time.time()-t0),flush=True)
    cds_store.append(cd)
pickle.dump(dict(oof=oof,abl=abl,oofp=oofp,cds=cds_store,thr=thr_res),open('exp_raw.pkl','wb'))
print('saved',time.time()-t0)
