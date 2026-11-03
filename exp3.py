import numpy as np, pickle, json, time
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from scipy.stats import spearmanr
from xgboost import XGBClassifier
from feats import feat14, parse, GC, seg_dist, FEATS, GROUPS
t0=time.time(); R,K,B=3,5,300
recs=pickle.load(open('records2.pkl','rb'))
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
        x=np.concatenate([base,[0],np.full(5,np.nan)]); rows.append(dict(m=r['match'],c=r['comp'],t=0,x=x,yA=r['y'],yB=r['y_goal'],xg=r['xg'],loc=loc,tm=tm,op=op,gk=gk,id=r['id']))
    else:
        end=np.array([r['end'][0]*SX,r['end'][1]*SY]); x=np.concatenate([base,[1],pfe(loc,end,oa)])
        rows.append(dict(m=r['match'],c=r['comp'],t=1,x=x,yA=r['y'],yB=r['y_goal'],ok=r['pass_ok'],end=end,loc=loc,tm=tm,op=op,gk=gk,id=r['id']))
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

W0=np.array([.25]*4)
def cand(loc,tm,op,gk):
    oa=opall_of(op,gk); base=feat14(loc,tm,op,gk); F=[];comp=[]
    for j in range(len(tm)):
        end=tm[j]; pf=pfe(loc,end,oa); F.append(np.concatenate([base,[1],pf]))
        comp.append([1-min(pf[0]/30,1),min((10 if np.isnan(pf[1]) else pf[1])/10,1),min((10 if np.isnan(pf[2]) else pf[2])/10,1),np.clip(pf[4]/20,0,1)])
    return np.array(F),np.array(comp)
Y=yB; f=fold_of(0); oof=np.zeros(len(Y)); cds=[]
for k in range(K):
    tr=f!=k; te=f==k; mdl=xgbm().fit(X[tr],Y[tr]); oof[te]=mdl.predict_proba(X[te])[:,1]
    ev=[i for i in sh if te[i] and len(rows[i]['tm'])>0]; Fl=[];meta=[]
    for i in ev:
        q=rows[i]; F,comp=cand(q['loc'],q['tm'],q['op'],q['gk']); Fl.append(F); meta.append((len(F),comp))
    pp=mdl.predict_proba(np.vstack(Fl))[:,1]; psh=mdl.predict_proba(X[ev])[:,1]; off=0
    for z,(nk,comp) in enumerate(meta):
        prk=pp[off:off+nk]; off+=nk; cds.append(prk[np.argmax(comp@W0)]-psh[z])
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(3.3,1.8),dpi=300)
a=ax[0]; a.plot([0,0.6],[0,0.6],'--',lw=0.6,color='gray')
for nm,ii,col in (('Shots',sh,'#2F4B7C'),('Passes',ps,'#C46A12')):
    p=oof[ii]; y=Y[ii]; edges=np.unique(np.quantile(p,np.linspace(0,1,9))); idx=np.clip(np.digitize(p,edges[1:-1]),0,len(edges)-2)
    xs=[p[idx==j].mean() for j in range(len(edges)-1)]; ys=[y[idx==j].mean() for j in range(len(edges)-1)]
    a.plot(xs,ys,'o-',ms=2.2,lw=0.8,color=col,label=nm)
a.set_xlabel('Predicted probability',fontsize=5.3); a.set_ylabel('Observed rate',fontsize=5.3); a.set_title('(a) Calibration (goal label)',fontsize=5.6); a.tick_params(labelsize=4.8); a.legend(fontsize=4.5,frameon=False,loc='upper left')
b=ax[1]; b.hist(cds,bins=40,color='#6A1B9A',edgecolor='white',lw=0.2); b.axvline(0,color='k',lw=0.6,ls='--')
b.set_xlabel('CDS (goal label)',fontsize=5.3); b.set_ylabel('Shots',fontsize=5.3); b.set_title('(b) CDS distribution',fontsize=5.6); b.tick_params(labelsize=4.8)
plt.tight_layout(pad=0.4); fig.savefig('fig_calibration_cds.png',dpi=300)
print(len(cds),float(np.mean(np.array(cds)>0)),float(np.mean(cds)))
