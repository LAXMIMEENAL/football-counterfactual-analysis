import numpy as np, pickle
SX,SY=105/120,68/80
GC=np.array([105.0,34.0]); PL=np.array([105.0,34-3.66]); PR=np.array([105.0,34+3.66])
FEATS=['player_x','player_y','distance_to_goal','shot_angle','goal_defender_distance','nearest_defender_distance','mean_defender_distance','farthest_defender_distance','defenders_ahead','defender_separation','shooting_corridor','nearest_teammate_distance','mean_teammate_distance','attacker_defender_advantage']
PFEATS=['pass_length','lane_clearance','receiver_freedom']
GROUPS={'Player Position':[0,1],'Goal Context':[2,3,4],'Defensive Context':[5,6,7,8,9,10],'Teammate Context':[11,12,13]}
def seg_dist(p,a,b):
    ab=b-a; L=ab@ab
    if L==0: return np.linalg.norm(p-a,axis=1)
    t=np.clip(((p-a)@ab)/L,0,1)
    return np.linalg.norm(p-(a+t[:,None]*ab),axis=1)
def in_tri(pts,a,b,c):
    def sg(p,q,r): return (p[:,0]-r[0])*(q[1]-r[1])-(q[0]-r[0])*(p[:,1]-r[1])
    d1=sg(pts,a,b);d2=sg(pts,b,c);d3=sg(pts,c,a)
    neg=(d1<0)|(d2<0)|(d3<0); pos=(d1>0)|(d2>0)|(d3>0)
    return ~(neg&pos)
def feat14(a,tm,op,gk):
    f=np.full(14,np.nan)
    f[0],f[1]=a
    f[2]=np.linalg.norm(a-GC)
    v1,v2=PL-a,PR-a
    f[3]=np.arccos(np.clip(v1@v2/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9),-1,1))
    if len(op):
        d=np.linalg.norm(op-a,axis=1)
        f[4]=np.linalg.norm(op-GC,axis=1).min()
        f[5],f[6],f[7]=d.min(),d.mean(),d.max()
        f[8]=(op[:,0]>a[0]).sum()
        if len(op)>=2:
            n3=op[np.argsort(d)[:3]]
            f[9]=np.mean([np.linalg.norm(n3[i]-n3[j]) for i in range(len(n3)) for j in range(i+1,len(n3))])
    blockers=op if gk is None else np.vstack([op,gk[None]]) if len(op) else gk[None]
    if len(blockers): f[10]=in_tri(blockers,a,PL,PR).sum()
    else: f[10]=0
    if len(tm):
        dt=np.linalg.norm(tm-a,axis=1); f[11]=dt.min(); f[12]=dt.mean()
    near_t=(np.linalg.norm(tm-a,axis=1)<10).sum() if len(tm) else 0
    near_o=(np.linalg.norm(op-a,axis=1)<10).sum() if len(op) else 0
    f[13]=near_t-near_o
    return f
def pass3(passer,end,opp_all):
    out=np.full(3,np.nan); out[0]=np.linalg.norm(end-passer)
    if len(opp_all):
        out[1]=seg_dist(opp_all,passer,end).min(); out[2]=np.linalg.norm(opp_all-end,axis=1).min()
    return out
def parse(r):
    ff=r['ff']; m=np.array([[x*SX,y*SY] for (t,k,a,x,y) in ff]) if ff else np.zeros((0,2))
    tmask=np.array([t for (t,k,a,x,y) in ff],bool); kmask=np.array([k for (t,k,a,x,y) in ff],bool); amask=np.array([a for (t,k,a,x,y) in ff],bool)
    tm=m[tmask&~amask] if len(m) else m
    opmask=~tmask&~kmask; op=m[opmask]; gkm=~tmask&kmask
    gk=m[gkm][0] if gkm.any() else None
    loc=np.array([r['loc'][0]*SX,r['loc'][1]*SY])
    return loc,tm,op,gk
def build(recs):
    S=[];P=[]
    for r in recs:
        loc,tm,op,gk=parse(r)
        opall=op if gk is None else (np.vstack([op,gk[None]]) if len(op) else gk[None])
        if r['type']=='Shot':
            S.append((r['match'],r['id'],r['y'],r['xg'],feat14(loc,tm,op,gk),loc,tm,op,gk,r['comp']))
        else:
            end=np.array([r['end'][0]*SX,r['end'][1]*SY])
            tm2=np.vstack([tm,loc[None]]) if len(tm) else loc[None]
            P.append((r['match'],r['id'],r['y'],np.concatenate([feat14(end,tm2,op,gk),pass3(loc,end,opall)]),r['pass_ok'],r['comp']))
    return S,P
if __name__=='__main__':
    recs=pickle.load(open('records.pkl','rb'))
    S,P=build(recs)
    pickle.dump((S,P),open('built.pkl','wb'))
    ys=np.array([s[2] for s in S]); yp=np.array([p[2] for p in P])
    print(len(S),ys.mean(),len(P),yp.mean(), np.mean([np.isnan(s[4]).any() for s in S]))
