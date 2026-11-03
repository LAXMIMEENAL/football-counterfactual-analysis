"""Part B: receiver-heuristic validation with paired CIs, threshold and visibility sensitivity, and a pass-completion feasibility term. Needs records3.pkl."""
from common5 import *
t0=time.time(); out={}
Y=np.array([q['yg'][10] for q in rows]); f0=fold_of(0)
COLS=list(range(14))+list(range(15,20))
okp=np.array([bool(rows[i]['ok']) for i in ps]).astype(int)
# ---- completion model, out-of-fold on passes (match-grouped), also kept per fold for candidate scoring
models={}; pc_oof=np.zeros(len(ps)); fps=f0[ps]
for k in range(K):
    tr=fps!=k; te=fps==k
    m=xgbm().fit(X[ps][tr][:,COLS],okp[tr]); models[k]=m; pc_oof[te]=m.predict_proba(X[ps][te][:,COLS])[:,1]
def ece(y,p,nb=10):
    b=np.minimum((p*nb).astype(int),nb-1); return sum((b==i).mean()*abs(y[b==i].mean()-p[b==i].mean()) for i in range(nb) if (b==i).any())
out['completion_model']=dict(rate=float(okp.mean()),auc=float(roc_auc_score(okp,pc_oof)),brier=float(brier_score_loss(okp,pc_oof)),ece=float(ece(okp,pc_oof)))
print(out['completion_model'],flush=True)
# ---- per-pass receiver records
recs=[]
for n,i in enumerate(ps):
    q=rows[i]
    if not q['ok'] or len(q['tm'])<2: continue
    dend=np.linalg.norm(q['tm']-q['end'],axis=1); j=int(np.argmin(dend))
    F,comp,base,pfs=cand_basic(q['loc'],q['tm'],q['op'],q['gk'])
    pcand=models[fps[n]].predict_proba(np.column_stack([np.tile(base,(len(F),1)),pfs]))[:,1]
    s_lane=comp.sum(1)*0.25; s_feas=0.2*(comp.sum(1)+pcand)
    near=int(np.argmin(np.linalg.norm(q['tm']-q['loc'],axis=1)))
    ol=np.argsort(-s_lane); of=np.argsort(-s_feas)
    recs.append(dict(m=q['m'],d=float(dend[j]),nvis=len(q['tm']),lane1=int(ol[0]==j),lane3=int(j in ol[:3]),feas1=int(of[0]==j),feas3=int(j in of[:3]),near1=int(near==j),pc1=int(np.argmax(pcand)==j),ch=1/len(q['tm'])))
print('receiver records',len(recs),round(time.time()-t0),flush=True)
M_=np.array([r['m'] for r in recs]); D=np.array([r['d'] for r in recs]); NV=np.array([r['nvis'] for r in recs])
A={k:np.array([r[k] for r in recs]) for k in ('lane1','lane3','feas1','feas3','near1','pc1','ch')}
um=np.unique(M_); mi={u:np.where(M_==u)[0] for u in um}; rng=np.random.default_rng(13)
draws=[np.concatenate([mi[u] for u in rng.choice(um,len(um))]) for _ in range(500)]
def paired(a,b,sel):
    idx=np.where(sel)[0]; pt=float(A[a][idx].mean()-A[b][idx].mean())
    bs=[]
    for d in draws:
        d=d[sel[d]]; bs.append(A[a][d].mean()-A[b][d].mean())
    return [pt,float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
main=D<=4.0
out['main_4m']=dict(n=int(main.sum()),lane1=float(A['lane1'][main].mean()),lane3=float(A['lane3'][main].mean()),near1=float(A['near1'][main].mean()),feas1=float(A['feas1'][main].mean()),feas3=float(A['feas3'][main].mean()),pc1=float(A['pc1'][main].mean()),chance=float(A['ch'][main].mean()),
   diff_lane_minus_near=paired('lane1','near1',main),diff_feas_minus_near=paired('feas1','near1',main),diff_feas_minus_lane=paired('feas1','lane1',main))
print(json.dumps(out['main_4m']),flush=True)
out['threshold']={}
for th in (2.0,3.0,4.0,5.0,6.0):
    s_=D<=th; out['threshold'][str(th)]=dict(n=int(s_.sum()),lane1=float(A['lane1'][s_].mean()),near1=float(A['near1'][s_].mean()),feas1=float(A['feas1'][s_].mean()),chance=float(A['ch'][s_].mean()),diff_lane_minus_near=paired('lane1','near1',s_))
print(json.dumps(out['threshold']),flush=True)
out['visibility']={}
for nm,lo,hi in (('2-3',2,3),('4-5',4,5),('6-8',6,8),('9+',9,99)):
    s_=main&(NV>=lo)&(NV<=hi)
    if s_.sum()>50: out['visibility'][nm]=dict(n=int(s_.sum()),lane1=float(A['lane1'][s_].mean()),near1=float(A['near1'][s_].mean()),feas1=float(A['feas1'][s_].mean()),chance=float(A['ch'][s_].mean()),diff_lane_minus_near=paired('lane1','near1',s_))
print(json.dumps(out['visibility']),flush=True)
# ---- CDS with feasibility-aware receiver selection (primary goal label, rep 0)
SH=[i for i in sh if len(rows[i]['tm'])>0]; cds_l={}; cds_f={}; psh={}
for k in range(K):
    tr=f0!=k; te=f0==k; um_=xgbm().fit(X[tr],Y[tr]); ev=[i for i in SH if te[i]]
    # completion model for this fold was trained on passes of training matches (same fold ids)
    C_=[cand_basic(rows[i]['loc'],rows[i]['tm'],rows[i]['op'],rows[i]['gk']) for i in ev]
    pp=um_.predict_proba(np.vstack([c[0] for c in C_]))[:,1]; ps_=um_.predict_proba(X[ev])[:,1]
    pc=models[k].predict_proba(np.vstack([np.column_stack([np.tile(c[2],(len(c[0]),1)),c[3]]) for c in C_]))[:,1]; off=0
    for z,i in enumerate(ev):
        nk=len(C_[z][0]); prk=pp[off:off+nk]; pck=pc[off:off+nk]; off+=nk
        a=int(np.argmax(C_[z][1].sum(1))); b=int(np.argmax(0.2*(C_[z][1].sum(1)+pck)))
        cds_l[i]=prk[a]-ps_[z]; cds_f[i]=prk[b]-ps_[z]; psh[i]=ps_[z]
vl=np.array([cds_l[i] for i in SH]); vf=np.array([cds_f[i] for i in SH])
out['cds_feasibility']=dict(n=len(SH),lane_mean=float(vl.mean()),lane_pos=float((vl>0).mean()),feas_mean=float(vf.mean()),feas_pos=float((vf>0).mean()),same_receiver_share=float(np.mean([True]) ) )
print(json.dumps(out['cds_feasibility']),flush=True)
json.dump(out,open('results5b.json','w'),indent=1); print('done',round(time.time()-t0))
