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
print(out,flush=True)
matches=np.array(sorted(set(M)))
def fold_of(rep):
    perm=np.random.default_rng(rep).permutation(matches); fm={m:i%K for i,m in enumerate(perm)}; return np.array([fm[m] for m in M])
def m4(y,p):
    b=np.minimum((p*10).astype(int),9); e=sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(10) if (b==i).any())
    return np.array([roc_auc_score(y,p),average_precision_score(y,p),brier_score_loss(y,p),e])
def pairs(a,b):  # candidate features for counterfactual passes
    pass
# ---------- 1) unified model CV + counterfactuals, for each label
W0=np.array([.25]*4); pert=[]
for i in range(4):
    for sg in (1,-1):
        w=W0.copy(); w[i]+=sg*0.1; pert.append(w/w.sum())
def cand(q,loc,tm,op,gk):
    oa=opall_of(op,gk); base=feat14(loc,tm,op,gk); F=[];comp=[]
    for j in range(len(tm)):
        end=tm[j]; pf=pfe(loc,end,oa); F.append(np.concatenate([base,[1],pf]))
        comp.append([1-min(pf[0]/30,1),min((10 if np.isnan(pf[1]) else pf[1])/10,1),min((10 if np.isnan(pf[2]) else pf[2])/10,1),np.clip(pf[4]/20,0,1)])
    return np.array(F),np.array(comp)
res={}
for lab,Y in (('A',yA),('B',yB)):
    oof=np.zeros((R,len(Y))); cds=[]
    for rep in range(R):
        f=fold_of(rep); cd={k:np.full(len(sh),np.nan) for k in ['ps','top1','t3min','t3max','near','wst','nst','pp1']}
        rng=np.random.default_rng(900+rep)
        for k in range(K):
            tr=f!=k; te=f==k
            mdl=xgbm().fit(X[tr],Y[tr]); oof[rep,te]=mdl.predict_proba(X[te])[:,1]
            ev=[n for n,i in enumerate(sh) if te[i] and len(rows[i]['tm'])>0]
            Fl=[];meta=[];Fn=[];nmeta=[];nshot=[]
            for n in ev:
                q=rows[sh[n]]; F,comp=cand(q,q['loc'],q['tm'],q['op'],q['gk']); Fl.append(F); meta.append((n,len(F),comp,np.linalg.norm(q['tm']-q['loc'],axis=1)))
                nl=q['loc']+rng.normal(0,1,2); ntm=q['tm']+rng.normal(0,1,q['tm'].shape); nop=q['op']+rng.normal(0,1,q['op'].shape) if len(q['op']) else q['op']; ngk=None if q['gk'] is None else q['gk']+rng.normal(0,1,2)
                Fn_,ncomp=cand(q,nl,ntm,nop,ngk); Fn.append(Fn_); nmeta.append(ncomp); nshot.append(np.concatenate([feat14(nl,ntm,nop,ngk),[0],np.full(5,np.nan)]))
            pp=mdl.predict_proba(np.vstack(Fl))[:,1]; psh=mdl.predict_proba(X[[sh[n] for n in ev]])[:,1]
            npp=mdl.predict_proba(np.vstack(Fn))[:,1]; npsh=mdl.predict_proba(np.array(nshot))[:,1]
            off=0;noff=0
            for z,(n,nk,comp,dist) in enumerate(meta):
                prk=pp[off:off+nk]; off+=nk; c=prk-psh[z]; order=np.argsort(-(comp@W0))
                cd['ps'][n]=psh[z]; cd['top1'][n]=c[order[0]]; cd['pp1'][n]=prk[order[0]]; t3=c[order[:3]]; cd['t3min'][n]=t3.min(); cd['t3max'][n]=t3.max(); cd['near'][n]=c[np.argmin(dist)]
                b0=np.sign(c[order[0]]); cd['wst'][n]=float(all(np.sign(c[np.argmax(comp@w)])==b0 for w in pert))
                ns=nmeta[z]@W0; npk=npp[noff:noff+len(ns)]; noff+=len(ns); cd['nst'][n]=float(np.sign(npk[np.argmax(ns)]-npsh[z])==b0)
        cds.append(cd); print(lab,'rep',rep,round(time.time()-t0),flush=True)
    # metrics
    rr={}
    idx={'all':np.arange(len(Y)),'shots':sh,'passes':ps}
    um=np.unique(M); mi={u:np.where(M==u)[0] for u in um}; rng=np.random.default_rng(5)
    for nm,ii in idx.items():
        pt=np.mean([m4(Y[ii],oof[r][ii]) for r in range(R)],axis=0)
        inset=np.zeros(len(Y),bool); inset[ii]=True; bs=[]
        for _ in range(B):
            sel=np.concatenate([mi[u] for u in rng.choice(um,len(um))]); sel=sel[inset[sel]]
            bs.append(np.mean([roc_auc_score(Y[sel],oof[r][sel]) for r in range(R)]))
        rr[nm]=dict(pt=pt.tolist(),auc_lo=float(np.percentile(bs,2.5)),auc_hi=float(np.percentile(bs,97.5)))
    ok=~np.isnan(cds[0]['top1']); per=lambda f: float(np.mean([f(c) for c in cds]))
    avg=lambda k: np.nanmean([c[k] for c in cds],axis=0)
    q=np.quantile(XG[ok],[.25,.75]); a1=avg('top1')
    rr['cds']=dict(n=int(ok.sum()),mean=per(lambda c:np.nanmean(c['top1'])),median=per(lambda c:np.nanmedian(c['top1'])),q1=per(lambda c:np.nanpercentile(c['top1'],25)),q3=per(lambda c:np.nanpercentile(c['top1'],75)),
        pos=per(lambda c:np.mean(c['top1'][ok]>0)),t3=per(lambda c:np.mean(((c['t3min'][ok]>0)&(c['t3max'][ok]>0))|((c['t3min'][ok]<0)&(c['t3max'][ok]<0)))),
        wst=per(lambda c:np.nanmean(c['wst'][ok])),nst=per(lambda c:np.nanmean(c['nst'][ok])),near_mean=per(lambda c:np.nanmean(c['near'])),near_pos=per(lambda c:np.mean(c['near'][ok]>0)),
        ps_mean=per(lambda c:np.nanmean(c['ps'])),pp_mean=per(lambda c:np.nanmean(c['pp1'])),
        rho_cds_xg=float(spearmanr(a1[ok],XG[ok])[0]),rho_ps_xg=float(spearmanr(avg('ps')[ok],XG[ok])[0]),
        low=float(np.nanmean(a1[ok&(XG<=q[0])])),high=float(np.nanmean(a1[ok&(XG>=q[1])])))
    res[lab]=rr; print(lab,json.dumps(rr['cds']),flush=True)
out['unified']=res
# ---------- 2) xG baseline + shot-only models (label A,B)
um=np.unique(M[sh]); Ms=M[sh]; mis={u:np.where(Ms==u)[0] for u in um}; rng=np.random.default_rng(11)
xg_res={}
Xs=X[sh][:,:14]
for lab,Y in (('A',yA[sh]),('B',yB[sh])):
    o14=np.zeros((R,len(sh))); o14x=np.zeros((R,len(sh)))
    for rep in range(R):
        f=fold_of(rep)[sh]
        for k in range(K):
            tr=f!=k; te=f==k
            o14[rep,te]=xgbm().fit(Xs[tr],Y[tr]).predict_proba(Xs[te])[:,1]
            Xx=np.column_stack([Xs,XG]); o14x[rep,te]=xgbm().fit(Xx[tr],Y[tr]).predict_proba(Xx[te])[:,1]
    sc={'xG alone':np.tile(XG,(R,1)),'14 features':o14,'14 features + xG':o14x}
    pts={k:np.mean([roc_auc_score(Y,v[r]) for r in range(R)]) for k,v in sc.items()}
    draws=[np.concatenate([mis[u] for u in rng.choice(um,len(um))]) for _ in range(B)]
    bs={k:np.array([np.mean([roc_auc_score(Y[d],v[r][d]) for r in range(R)]) for d in draws]) for k,v in sc.items()}
    xg_res[lab]={k:dict(auc=float(pts[k]),lo=float(np.percentile(bs[k],2.5)),hi=float(np.percentile(bs[k],97.5)),prc=float(np.mean([average_precision_score(Y,v[r]) for r in range(R)]))) for k,v in sc.items()}
    for a,b in (('14 features + xG','xG alone'),('xG alone','14 features'),('14 features + xG','14 features')):
        d=bs[a]-bs[b]; xg_res[lab]['diff: %s - %s'%(a,b)]=[float(pts[a]-pts[b]),float(np.percentile(d,2.5)),float(np.percentile(d,97.5))]
    print('xg',lab,json.dumps({k:v for k,v in xg_res[lab].items() if not k.startswith('diff')}),flush=True)
out['xg']=xg_res
# ---------- 3) ablation CIs (label A, from earlier OOF), paired bootstrap
d0=pickle.load(open('exp_raw.pkl','rb')); S,P=pickle.load(open('built.pkl','rb'))
yS=np.array([s[2] for s in S]); mS=np.array([s[0] for s in S]); ums=np.unique(mS); miS={u:np.where(mS==u)[0] for u in ums}; rng=np.random.default_rng(21)
draws=[np.concatenate([miS[u] for u in rng.choice(ums,len(ums))]) for _ in range(B)]
full=d0['oof']['XGB']; ab={}
for g in GROUPS:
    pt=np.mean([roc_auc_score(yS,d0['abl'][g][r])-roc_auc_score(yS,full[r]) for r in range(R)])
    bs=[np.mean([roc_auc_score(yS[d],d0['abl'][g][r][d])-roc_auc_score(yS[d],full[r][d]) for r in range(R)]) for d in draws]
    ab[g]=[float(pt),float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
out['ablation_delta']=ab; print('abl',ab,flush=True)
# ---------- 4) leave-one-competition-out (shots, labels A,B)
loco={}; Cs=C[sh]
for lab,Y in (('A',yA[sh]),('B',yB[sh])):
    loco[lab]={}; pooled=np.zeros(len(sh))
    for c in np.unique(Cs):
        te=Cs==c; mdl=xgbm().fit(Xs[~te],Y[~te]); p=mdl.predict_proba(Xs[te])[:,1]; pooled[te]=p
        mm=np.unique(Ms[te]); mi2={u:np.where((Ms==u)&te)[0] for u in mm}; rg=np.random.default_rng(31); bsl=[]
        for _ in range(200):
            s_=np.concatenate([mi2[u] for u in rg.choice(mm,len(mm))]); bsl.append(roc_auc_score(Y[s_],pooled_tmp[s_]) if False else 0)
        loco[lab][c]=dict(auc=float(roc_auc_score(Y[te],p)),n=int(te.sum()))
    # bootstrap CIs per competition
    for c in np.unique(Cs):
        te=np.where(Cs==c)[0]; mm=np.unique(Ms[te]); mi2={u:np.where((Ms==u)&(Cs==c))[0] for u in mm}; rg=np.random.default_rng(31)
        bsl=[roc_auc_score(Y[s_],pooled[s_]) for s_ in (np.concatenate([mi2[u] for u in rg.choice(mm,len(mm))]) for _ in range(200))]
        loco[lab][c]['lo']=float(np.percentile(bsl,2.5)); loco[lab][c]['hi']=float(np.percentile(bsl,97.5))
    loco[lab]['mean']=float(np.mean([v['auc'] for k,v in loco[lab].items() if isinstance(v,dict)]))
out['loco']=loco; print('loco',json.dumps(loco),flush=True)
# ---------- 5) receiver-heuristic validation on observed completed passes
tp1=tpn=tp3=tpr=0; n=0; chance=[]; rng=np.random.default_rng(41)
for i in ps:
    q=rows[i]
    if not q['ok'] or len(q['tm'])<2: continue
    dend=np.linalg.norm(q['tm']-q['end'],axis=1); j=int(np.argmin(dend))
    if dend[j]>4.0: continue
    oa=opall_of(q['op'],q['gk']); comp=[]
    for t in q['tm']:
        pf=pfe(q['loc'],t,oa); comp.append([1-min(pf[0]/30,1),min((10 if np.isnan(pf[1]) else pf[1])/10,1),min((10 if np.isnan(pf[2]) else pf[2])/10,1),np.clip(pf[4]/20,0,1)])
    sc=np.array(comp)@W0; order=np.argsort(-sc); near=np.argmin(np.linalg.norm(q['tm']-q['loc'],axis=1))
    n+=1; tp1+=order[0]==j; tp3+=j in order[:3]; tpn+=near==j; chance.append(1/len(q['tm']))
out['receiver']=dict(n=n,top1_lane=tp1/n,top3_lane=tp3/n,top1_nearest=tpn/n,chance_top1=float(np.mean(chance)))
print('receiver',out['receiver'],flush=True)
json.dump(out,open('results2.json','w'),indent=1); print('done',round(time.time()-t0))
