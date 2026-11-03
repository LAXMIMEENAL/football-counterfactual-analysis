"""Compares numbers printed in the paper with the result files. Prints MISMATCH lines, otherwise OK."""
import json
r1=json.load(open('results.json')); r2=json.load(open('results2.json')); e=json.load(open('err.json')); sh=json.load(open('shap2.json'))
chk=[]
def c(name,val,paper,tol=0.0006): chk.append((name,round(val,3),paper,abs(val-paper)<=tol))
for k,(a,b,cc,dd) in {'LR':(0.626,0.487,0.215,0.019),'RF':(0.623,0.484,0.216,0.024),'XGB-pilot':(0.627,0.492,0.219,0.071),'XGB':(0.624,0.488,0.216,0.025)}.items():
    v=r1['models'][k]['pt']; c(k+' auc',v[0],a); c(k+' prauc',v[1],b); c(k+' brier',v[2],cc); c(k+' ece',v[3],dd)
for lab,(xa,f14,fx) in {'A':(0.628,0.624,0.650),'B':(0.794,0.745,0.780)}.items():
    c('xG '+lab,r2['xg'][lab]['xG alone']['auc'],xa); c('14f '+lab,r2['xg'][lab]['14 features']['auc'],f14); c('14f+xG '+lab,r2['xg'][lab]['14 features + xG']['auc'],fx)
for lab,(s,p,mean,pos,rho) in {'A':(0.629,0.725,-0.216,0.001,-0.27),'B':(0.754,0.756,-0.051,0.140,-0.70)}.items():
    u=r2['unified'][lab]; c('uni shots '+lab,u['shots']['pt'][0],s); c('uni pass '+lab,u['passes']['pt'][0],p); c('mean cds '+lab,u['cds']['mean'],mean); c('pos '+lab,u['cds']['pos'],pos,0.001); c('rho '+lab,u['cds']['rho_cds_xg'],rho,0.006)
rc=r2['receiver']; c('top1 lane',rc['top1_lane'],0.464,0.001); c('top1 near',rc['top1_nearest'],0.471,0.001); c('top3',rc['top3_lane'],0.845,0.001); c('chance',rc['chance_top1'],0.174,0.001); c('n recv',rc['n'],20056,0)
for g,v in {'Goal Context':-0.008,'Defensive Context':-0.014}.items(): c('abl '+g,r2['ablation_delta'][g][0],v,0.0006)
c('loco A',r2['loco']['A']['mean'],0.625); c('loco B',r2['loco']['B']['mean'],0.749)
for row,v in zip(e,(0.657,0.612,0.555)): c('terc '+row[0],row[3],v)
c('shap B top',sh['B']['top'][0][1],0.508); c('shap A top',sh['A']['top'][0][1],0.220)
import os
if os.path.exists('results4.json'):
    r4=json.load(open('results4.json'))
    for nm,(ps_,pp_,cd_,pos_,rho_) in {'goal<=5s':(0.107,0.052,-0.055,0.096,-0.68),'goal<=10s':(0.109,0.057,-0.052,0.142,-0.69),'goal<=15s':(0.111,0.061,-0.050,0.147,-0.71),'goal<=20s':(0.111,0.064,-0.047,0.170,-0.72),'goal<=30s':(0.111,0.067,-0.044,0.198,-0.73),'downstream-only (excl. own goal), 10s':(0.014,0.048,0.034,0.907,0.23)}.items():
        v=r4[nm]; c('t8 pshot '+nm,v['mean_pshot'],ps_); c('t8 ppass '+nm,v['mean_ppass'],pp_); c('t8 cds '+nm,v['mean_cds'],cd_); c('t8 pos '+nm,v['pos'],pos_,0.0006); c('t8 rho '+nm,v['rho_xg'],rho_,0.006)
    c('own goal share',r4['own_goal_share_of_shot_label_B'],0.875,0.0006)
bad=[x for x in chk if not x[3]]; print('checked',len(chk),'values;',('MISMATCH:' if bad else 'all match'),bad)
# --- sensitivity analysis (Table VIII), results4.json from exp4.py
import os
if os.path.exists('results4.json'):
    r4=json.load(open('results4.json')); chk=[]
    rows={'goal<=5s':(0.107,0.052,-0.055,0.096,-0.68),'goal<=10s':(0.109,0.057,-0.052,0.142,-0.69),'goal<=15s':(0.111,0.061,-0.050,0.147,-0.71),
          'goal<=20s':(0.111,0.064,-0.047,0.170,-0.72),'goal<=30s':(0.111,0.067,-0.044,0.198,-0.73),'downstream-only (excl. own goal), 10s':(0.014,0.048,0.034,0.907,0.23)}
    for k,(ps,pp,m,pos,rho) in rows.items():
        v=r4[k]; c(k+' pshot',v['mean_pshot'],ps,0.0006); c(k+' ppass',v['mean_ppass'],pp,0.0006); c(k+' cds',v['mean_cds'],m,0.0006); c(k+' pos',v['pos'],pos,0.0006); c(k+' rho',v['rho_xg'],rho,0.006)
    c('own goal share',r4['own_goal_share_of_shot_label_B'],0.875,0.001)
    bad=[x for x in chk if not x[3]]; print('sensitivity: checked',len(chk),'values;',('MISMATCH:' if bad else 'all match'),bad)
