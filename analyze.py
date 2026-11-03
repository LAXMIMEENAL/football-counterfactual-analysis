import numpy as np, pickle, json
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from feats import *
d=pickle.load(open('exp_raw.pkl','rb')); S,P=pickle.load(open('built.pkl','rb'))
ys=np.array([s[2] for s in S]); ms=np.array([s[0] for s in S]); xg=np.array([s[3] for s in S]); Xs=np.array([s[4] for s in S])
yp=np.array([p[2] for p in P]); mp=np.array([p[0] for p in P])
R=3; B=500; rng=np.random.default_rng(7)
def ece(y,p,nb=10):
    b=np.minimum((p*nb).astype(int),nb-1); return sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(nb) if (b==i).any())
def m4(y,p): return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),ece(y,p)])
def boot(y,m,preds):  # preds: R x N
    um=np.unique(m); idx={u:np.where(m==u)[0] for u in um}
    pt=np.mean([m4(y,preds[r]) for r in range(R)],axis=0); bs=[]
    for _ in range(B):
        sel=np.concatenate([idx[u] for u in rng.choice(um,len(um))])
        bs.append(np.mean([m4(y[sel],preds[r][sel]) for r in range(R)],axis=0))
    bs=np.array(bs); return pt,np.percentile(bs,2.5,axis=0),np.percentile(bs,97.5,axis=0),bs
out={}
out['N']=dict(shots=len(ys),passes=len(yp),matches=int(len(set(ms)|set(mp))),shot_pos=float(ys.mean()),pass_pos=float(yp.mean()),nogk=int(sum(s[8] is None for s in S)))
res={}
for name in ['LR','RF','XGB-pilot','XGB']:
    pt,lo,hi,bs=boot(ys,ms,d['oof'][name]); res[name]=(pt,lo,hi,bs); print(name,np.round(pt,3),np.round(lo,3),np.round(hi,3),flush=True)
out['models']={k:dict(pt=v[0].tolist(),lo=v[1].tolist(),hi=v[2].tolist()) for k,v in res.items()}
# paired difference XGB - LR AUC (same bootstrap draws not shared; recompute paired)
um=np.unique(ms); idx={u:np.where(ms==u)[0] for u in um}; diffs=[]
for _ in range(B):
    sel=np.concatenate([idx[u] for u in rng.choice(um,len(um))])
    diffs.append(np.mean([roc_auc_score(ys[sel],d['oof']['XGB'][r][sel])-roc_auc_score(ys[sel],d['oof']['LR'][r][sel]) for r in range(R)]))
out['diff_xgb_lr']=[float(np.mean(diffs)),float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5))]
print('diff xgb-lr',out['diff_xgb_lr'])
# threshold metrics
thr=np.array(d['thr']); out['thr']=thr.mean(axis=0).tolist(); out['thr_sd']=thr.std(axis=0).tolist(); print('thr',np.round(thr.mean(0),3))
# logloss
out['logloss_xgb']=float(np.mean([log_loss(ys,d['oof']['XGB'][r]) for r in range(R)]))
# fold-level AUC sd (per rep overall) 
out['auc_rep_sd']=float(np.std([roc_auc_score(ys,d['oof']['XGB'][r]) for r in range(R)]))
# ablation
full=np.mean([roc_auc_score(ys,d['oof']['XGB'][r]) for r in range(R)]); out['abl']={'full':float(full)}
for g in GROUPS: out['abl'][g]=float(np.mean([roc_auc_score(ys,d['abl'][g][r]) for r in range(R)]))
print(out['abl'])
# pass model
resp={}
for name in ['XGB-17','XGB-14']:
    pt,lo,hi,_=boot(yp,mp,d['oofp'][name]); resp[name]=dict(pt=pt.tolist(),lo=lo.tolist(),hi=hi.tolist()); print(name,np.round(pt,3),np.round(lo,3),np.round(hi,3),flush=True)
out['pass']=resp
# CDS
cds=d['cds']; keys=['pshot','top1','top3min','top3max','near','wstab','nstab','ppass1','pnear']
valid=~np.isnan(cds[0]['top1']); out['cds_n']=int(valid.sum())
avg={k:np.nanmean([c[k] for c in cds],axis=0) for k in keys}
per=lambda f: float(np.mean([f(c) for c in cds]))
out['cds']=dict(
 mean=per(lambda c:np.nanmean(c['top1'])), median=per(lambda c:np.nanmedian(c['top1'])),
 q1=per(lambda c:np.nanpercentile(c['top1'],25)), q3=per(lambda c:np.nanpercentile(c['top1'],75)),
 pos=per(lambda c:np.nanmean(c['top1'][valid]>0)),
 top3_stable=per(lambda c:np.mean(((c['top3min'][valid]>0)&(c['top3max'][valid]>0))|((c['top3min'][valid]<0)&(c['top3max'][valid]<0)))),
 w_stable=per(lambda c:np.nanmean(c['wstab'][valid])), n_stable=per(lambda c:np.nanmean(c['nstab'][valid])),
 near_mean=per(lambda c:np.nanmean(c['near'])), near_pos=per(lambda c:np.nanmean(c['near'][valid]>0)),
 ppass_lane=per(lambda c:np.nanmean(c['ppass1'])), ppass_near=per(lambda c:np.nanmean(c['pnear'])),
 pshot_mean=per(lambda c:np.nanmean(c['pshot'])))
rho=spearmanr(avg['pshot'][valid],xg[valid]); out['rho_pshot_xg']=[float(rho[0])]
rho2=spearmanr(avg['top1'][valid],xg[valid]); out['rho_cds_xg']=[float(rho2[0])]
q=np.quantile(xg[valid],[.25,.75]); lowx=valid&(xg<=q[0]); highx=valid&(xg>=q[1])
out['cds_lowxg']=float(np.nanmean(avg['top1'][lowx])); out['cds_highxg']=float(np.nanmean(avg['top1'][highx])); out['pos_lowxg']=float(np.mean(avg['top1'][lowx]>0)); out['pos_highxg']=float(np.mean(avg['top1'][highx]>0))
print(json.dumps(out['cds'],indent=1)); print(out['rho_pshot_xg'],out['rho_cds_xg'],out['cds_lowxg'],out['cds_highxg'],out['pos_lowxg'],out['pos_highxg'])
# SHAP (pred_contribs) global on full data
m=XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,min_child_weight=5,subsample=0.8,colsample_bytree=0.75,reg_alpha=0.2,reg_lambda=5.0,eval_metric='logloss',random_state=42,n_jobs=1,tree_method='hist').fit(Xs,ys)
c=m.get_booster().predict(__import__('xgboost').DMatrix(Xs),pred_contribs=True)[:,:-1]; ma=np.abs(c).mean(0)
order=np.argsort(-ma); out['shap']=[(FEATS[i],float(ma[i])) for i in order[:5]]
out['shap_groups']={g:float(ma[idx].sum()/ma.sum()) for g,idx in GROUPS.items()}
print(out['shap'],out['shap_groups'])
json.dump(out,open('results.json','w'),indent=1)
