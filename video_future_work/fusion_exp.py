"""Fusion experiment on the video-paired subset. Run AFTER video_branch.py.
Usage: python fusion_exp.py --built built.pkl --video video_features.csv --out results_video.json
built.pkl comes from feats.py (StatsBomb shots with the 14 CSV features). Output fills Table IV (video/fusion rows), Table V, video ablations, video SHAP share."""
import argparse, json, pickle, numpy as np, pandas as pd, xgboost as xgb_
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
VF=['attacker_speed','attacker_heading','closing_speed','mean_defender_speed','local_pressure','ball_speed','camera_motion']
VG={'Attacker dynamics':[0,1],'Defender dynamics':[2,3,4],'Ball speed':[5],'Camera motion':[6]}
R,K,B=3,5,500
def mk(): return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,eval_metric='logloss',random_state=42,n_jobs=1,tree_method='hist')
def ece(y,p,nb=10):
    b=np.minimum((p*nb).astype(int),nb-1); return sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(nb) if (b==i).any())
def m4(y,p): return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),ece(y,p)])
def main(a):
    S,_=pickle.load(open(a.built,'rb')); vid=pd.read_csv(a.video)
    allrows=len(vid); carrier=vid['carrier_found'].fillna(False).astype(bool) if 'carrier_found' in vid else pd.Series(False,index=vid.index)
    ok=vid[vid['ok'].fillna(False).astype(bool)].set_index('event_id'); ids=[s[1] for s in S]
    sel=[i for i,e in enumerate(ids) if e in ok.index]
    y=np.array([S[i][2] for i in sel]); m=np.array([S[i][0] for i in sel]); Xc=np.array([S[i][4] for i in sel]); Xv=ok.loc[[ids[i] for i in sel],VF].to_numpy(float)
    N=len(y); um=np.unique(m); print('paired events',N,'matches',len(um),'pos rate',y.mean())
    Xe=np.hstack([Xc,Xv]); oof={k:np.zeros((R,N)) for k in ['csv','video','early','late']}; abl={g:np.zeros((R,N)) for g in VG}; alphas=[]
    for r in range(R):
        perm=np.random.default_rng(r).permutation(um); fm={u:i%K for i,u in enumerate(perm)}; f=np.array([fm[x] for x in m])
        for k in range(K):
            tr,te=f!=k,f==k
            if te.sum()==0 or tr.sum()==0: continue
            mc=mk().fit(Xc[tr],y[tr]); mv=mk().fit(Xv[tr],y[tr]); me=mk().fit(Xe[tr],y[tr])
            pc,pv,pe=[q.predict_proba(X[te])[:,1] for q,X in ((mc,Xc),(mv,Xv),(me,Xe))]
            oof['csv'][r,te],oof['video'][r,te],oof['early'][r,te]=pc,pv,pe
            # late fusion: alpha picked on inner 3-fold OOF of the training part
            tm=sorted(set(m[tr])); inner={u:i%3 for i,u in enumerate(np.random.default_rng(50+r*10+k).permutation(tm))}; fi=np.array([inner[u] for u in m[tr]])
            ic=np.zeros(tr.sum()); iv=np.zeros(tr.sum()); yt=y[tr]
            for j in range(3):
                ic[fi==j]=mk().fit(Xc[tr][fi!=j],yt[fi!=j]).predict_proba(Xc[tr][fi==j])[:,1]; iv[fi==j]=mk().fit(Xv[tr][fi!=j],yt[fi!=j]).predict_proba(Xv[tr][fi==j])[:,1]
            al=max(np.linspace(0,1,11),key=lambda x:roc_auc_score(yt,x*ic+(1-x)*iv)); alphas.append(al)
            oof['late'][r,te]=al*pc+(1-al)*pv
            for g,idx in VG.items():
                keep=[i for i in range(Xe.shape[1]) if i-14 not in idx or i<14]
                abl[g][r,te]=mk().fit(Xe[tr][:,keep],y[tr]).predict_proba(Xe[te][:,keep])[:,1]
    idxm={u:np.where(m==u)[0] for u in um}; rng=np.random.default_rng(7)
    res={'N_events':int(N),'N_matches':int(len(um)),'alpha_mean':float(np.mean(alphas))}
    pt={k:np.mean([m4(y,v[r]) for r in range(R)],axis=0) for k,v in oof.items()}
    draws=[np.concatenate([idxm[u] for u in rng.choice(um,len(um))]) for _ in range(B)]
    for k,v in oof.items():
        bs=np.array([np.mean([m4(y[s],v[r][s]) for r in range(R)],axis=0) for s in draws])
        res[k]=dict(pt=pt[k].tolist(),lo=np.percentile(bs,2.5,axis=0).tolist(),hi=np.percentile(bs,97.5,axis=0).tolist())
    for k in ('early','late'):
        d=[np.mean([roc_auc_score(y[s],oof[k][r][s])-roc_auc_score(y[s],oof['csv'][r][s]) for r in range(R)]) for s in draws]
        res['diff_'+k+'_vs_csv']=[float(np.mean(d)),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
    res['abl']={'early_full':float(np.mean([roc_auc_score(y,oof['early'][r]) for r in range(R)]))}
    for g in VG: res['abl'][g]=float(np.mean([roc_auc_score(y,abl[g][r]) for r in range(R)]))
    c=mk().fit(Xe,y).get_booster().predict(xgb_.DMatrix(Xe),pred_contribs=True)[:,:-1]; ma=np.abs(c).mean(0)
    res['shap_video_share']=float(ma[14:].sum()/ma.sum()); res['shap_video_top']=[(VF[i],float(ma[14+i])) for i in np.argsort(-ma[14:])[:3]]
    mr=vid['match_rate'].dropna(); pe=vid.loc[vid['ok'].fillna(False).astype(bool),'pos_err']
    res['table5']=dict(clips=int(allrows),carrier_found=float(carrier.mean()),match_rate=float(mr.mean()) if len(mr) else None,
                       pos_err=float(pe.mean()) if len(pe) else None,retained=int(len(ok)))
    json.dump(res,open(a.out,'w'),indent=1); print(json.dumps(res,indent=1))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--built',default='built.pkl'); ap.add_argument('--video',default='video_features.csv'); ap.add_argument('--out',default='results_video.json'); main(ap.parse_args())
