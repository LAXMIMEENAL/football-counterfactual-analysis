"""Video branch: clip -> pitch-referenced player tracks -> dynamic features.
Usage: python video_branch.py --manifest manifest.csv --records records.pkl --out video_features.csv
manifest.csv columns: event_id, video_path, t_event (seconds in the video at the decision frame), homography_json
homography_json: {"H": [[...],[...],[...]]} = 3x3 matrix mapping DECISION-FRAME image pixels -> pitch metres (105 x 68).
  Get it from SoccerNet camera calibration or by annotating >=4 pitch-line points on the decision frame (cv2.findHomography).
records.pkl: produced by fetch.py (StatsBomb events + 360 freeze frames), used only for alignment and validation.
Requires: pip install ultralytics opencv-python scipy scikit-learn pandas numpy
"""
import argparse, json, pickle, numpy as np, pandas as pd, cv2
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
SX,SY=105/120,68/80; GC=np.array([105.0,34.0])
T_SEC,FPS_USE,WIN_SEC,TAU,MATCH_M=3.0,10,0.5,2.5,3.0

def to_pitch(H,pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1,1,2),H).reshape(-1,2)

def read_frames(path,t_event):
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 25.0
    step=max(int(round(fps/FPS_USE)),1); end=int(round(t_event*fps)); start=max(end-int(T_SEC*fps),0)
    frames=[]
    for fi in range(start,end+1,step):
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi); ok,fr=cap.read()
        if ok: frames.append(fr)
    cap.release(); return frames, fps/step

def chain_to_last(frames,boxes_per_frame):
    """3x3 homographies mapping each frame's pixels into the LAST frame (background ORB matching, players masked)."""
    orb=cv2.ORB_create(1500); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True); n=len(frames)
    gray=[cv2.cvtColor(f,cv2.COLOR_BGR2GRAY) for f in frames]; Ms=[np.eye(3)]*(n-1); ok_all=True
    for t in range(n-1):
        masks=[]
        for g,bx in ((gray[t],boxes_per_frame[t]),(gray[t+1],boxes_per_frame[t+1])):
            m=np.full(g.shape,255,np.uint8)
            for (x1,y1,x2,y2) in bx: m[max(int(y1)-5,0):int(y2)+5,max(int(x1)-5,0):int(x2)+5]=0
            masks.append(m)
        k1,d1=orb.detectAndCompute(gray[t],masks[0]); k2,d2=orb.detectAndCompute(gray[t+1],masks[1])
        M=np.eye(3)
        if d1 is not None and d2 is not None and len(k1)>20 and len(k2)>20:
            mt=bf.match(d1,d2)
            if len(mt)>=15:
                a=np.float32([k1[m.queryIdx].pt for m in mt]); b=np.float32([k2[m.trainIdx].pt for m in mt])
                H,_=cv2.findHomography(a,b,cv2.RANSAC,3.0)
                if H is not None: M=H
                else: ok_all=False
            else: ok_all=False
        else: ok_all=False
        Ms[t]=M
    T=[np.eye(3)]*n; acc=np.eye(3)
    for t in range(n-2,-1,-1):
        acc=acc@Ms[t]; T[t]=acc.copy()
    return T,ok_all

def camera_motion(frames):
    g=[cv2.cvtColor(cv2.resize(f,None,fx=0.25,fy=0.25),cv2.COLOR_BGR2GRAY) for f in frames[-2:]]
    fl=cv2.calcOpticalFlowFarneback(g[0],g[1],None,0.5,3,15,3,5,1.2,0)
    return float(np.linalg.norm(fl,axis=2).mean())

def speed(track_pos,fps):
    k=max(int(round(WIN_SEC*fps)),1)
    if len(track_pos)<=k: return np.nan,None
    v=(track_pos[-1]-track_pos[-1-k])/(k/fps); s=float(np.linalg.norm(v))
    return (s if s<=12 else np.nan),v

def process(model,row,rec):
    frames,fps=read_frames(row.video_path,row.t_event)
    if len(frames)<4: return None
    H=np.array(json.load(open(row.homography_json))['H'],float)
    tracks={}; ball=[]; boxes=[]; crops={}
    for fi,fr in enumerate(frames):
        r=model.track(fr,persist=True,tracker='bytetrack.yaml',classes=[0,32],conf=0.25,verbose=False)[0]
        bx=[]
        if r.boxes is not None and r.boxes.id is not None:
            for b,i,c in zip(r.boxes.xyxy.cpu().numpy(),r.boxes.id.cpu().numpy().astype(int),r.boxes.cls.cpu().numpy().astype(int)):
                if c==0:
                    bx.append(b); tracks.setdefault(i,{})[fi]=np.array([(b[0]+b[2])/2,b[3]])
                    if fi==len(frames)-1:
                        x1,y1,x2,y2=[int(v) for v in b]; ch=fr[y1+(y2-y1)//5:y1+(y2-y1)//2,x1+(x2-x1)//4:x2-(x2-x1)//4]
                        if ch.size: crops[i]=cv2.cvtColor(ch,cv2.COLOR_BGR2HSV).reshape(-1,3).mean(0)
                elif c==32: ball.append((fi,np.array([(b[0]+b[2])/2,(b[1]+b[3])/2])))
        boxes.append(bx)
    T,reg_ok=chain_to_last(frames,boxes)
    last=len(frames)-1
    def P(fi,px):  # frame pixel -> pitch metres (before orientation fix)
        q=cv2.perspectiveTransform(np.float32([[px]]),T[fi])[0,0]; return to_pitch(H,[q])[0]
    pos={i:{fi:P(fi,p) for fi,p in d.items()} for i,d in tracks.items()}
    cur=[i for i in pos if last in pos[i]]
    if len(cur)<6: return None
    cur_xy=np.array([pos[i][last] for i in cur])
    # --- alignment with CSV freeze frame (also gives validation numbers); try both pitch orientations
    csv=np.array([[x*SX,y*SY] for (t,k,a,x,y) in rec['ff']]+[[rec['loc'][0]*SX,rec['loc'][1]*SY]])
    best=None
    for flip in (False,True):
        xy=np.array([105-cur_xy[:,0],68-cur_xy[:,1]]).T if flip else cur_xy
        D=np.linalg.norm(xy[:,None]-csv[None],axis=2); ri,ci=linear_sum_assignment(D)
        d=D[ri,ci]; m=d<MATCH_M
        sc=(d[m].mean() if m.any() else 99)-0.1*m.sum()
        if best is None or sc<best[0]: best=(sc,flip,ri,ci,d,m)
    _,flip,ri,ci,d,m=best
    fix=(lambda p: np.array([105-p[0],68-p[1]])) if flip else (lambda p:p)
    carrier_csv=len(csv)-1; mc=[(r_,c_) for r_,c_,mm in zip(ri,ci,m) if mm and c_==carrier_csv]
    out=dict(match_rate=float(m.sum()/len(csv)),pos_err=float(d[m].mean()) if m.any() else np.nan,carrier_found=bool(mc),camera_ok=bool(reg_ok))
    if not mc: out['ok']=False; return out
    car=cur[mc[0][0]]
    # --- team assignment: 2-means on torso colour; attackers = cluster of the carrier
    ids=[i for i in cur if i in crops]; km=KMeans(2,n_init=5,random_state=0).fit(np.array([crops[i] for i in ids]))
    lab=dict(zip(ids,km.labels_)); 
    if car not in lab: out['ok']=False; return out
    att=[i for i in ids if lab[i]==lab[car] and i!=car]; dfn=[i for i in ids if lab[i]!=lab[car]]
    def path(i): 
        fs=sorted(pos[i]); return np.array([fix(pos[i][f]) for f in fs])
    sc,vc=speed(path(car),fps); cp=fix(pos[car][last])
    head=np.nan
    if vc is not None and np.linalg.norm(vc)>0.3:
        g=(GC-cp)/np.linalg.norm(GC-cp); head=float(np.arccos(np.clip(vc@g/np.linalg.norm(vc),-1,1)))
    dpos=np.array([fix(pos[i][last]) for i in dfn]) if dfn else np.zeros((0,2))
    closing=np.nan
    if len(dpos):
        j=dfn[int(np.argmin(np.linalg.norm(dpos-cp,axis=1)))]; pj=path(j); pc=path(car); k=max(int(round(WIN_SEC*fps)),1)
        if len(pj)>k and len(pc)>k: closing=float((np.linalg.norm(pj[-1-k]-pc[-1-k])-np.linalg.norm(pj[-1]-pc[-1]))/(k/fps))
    ds=[speed(path(i),fps)[0] for i in dfn]; ds=[x for x in ds if not np.isnan(x)]
    bsp=np.nan
    if len(ball)>=3:
        bp=np.array([fix(P(fi,p)) for fi,p in ball[-4:]]); bsp=float(min(np.linalg.norm(bp[-1]-bp[0])/((ball[-1][0]-ball[-4][0])/fps),40))
    out.update(attacker_speed=sc,attacker_heading=head,closing_speed=closing,mean_defender_speed=float(np.mean(ds)) if ds else np.nan,
               local_pressure=float((np.linalg.norm(dpos-cp,axis=1)<5).sum()) if len(dpos) else 0.0,ball_speed=bsp,camera_motion=camera_motion(frames))
    out['ok']=bool(out['pos_err']<=TAU and out['camera_ok']); return out

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--records',required=True); ap.add_argument('--out',default='video_features.csv'); ap.add_argument('--weights',default='yolov8m.pt')
    a=ap.parse_args(); from ultralytics import YOLO
    model=YOLO(a.weights); man=pd.read_csv(a.manifest); recs={r['id']:r for r in pickle.load(open(a.records,'rb'))}; rows=[]
    for row in man.itertuples():
        try:
            o=process(model,row,recs[row.event_id]); 
        except Exception as e: print('fail',row.event_id,e); o=None
        if o is None: o=dict(ok=False)
        o['event_id']=row.event_id; rows.append(o); print(row.event_id,o.get('ok'),flush=True)
    pd.DataFrame(rows).to_csv(a.out,index=False)
