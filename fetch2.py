import json, requests, pickle, sys
from concurrent.futures import ThreadPoolExecutor
RAW='https://raw.githubusercontent.com/statsbomb/open-data/master/data'
COMPS=[(43,106,'WC2022'),(55,43,'Euro2020'),(55,282,'Euro2024'),(53,106,'WEuro2022'),(72,107,'WWC2023'),(1267,107,'AFCON2023'),(9,281,'Bundesliga2324')]
s=requests.Session()
def get(u):
    for _ in range(3):
        try:
            r=s.get(u,timeout=60)
            if r.status_code==200: return r.json()
            if r.status_code==404: return None
        except Exception as e: pass
    return None
def proc(args):
    mid,comp=args
    ev=get(f'{RAW}/events/{mid}.json'); tt=get(f'{RAW}/three-sixty/{mid}.json')
    if not ev or not tt: return None
    ff={f['event_uuid']:f['freeze_frame'] for f in tt}
    shots=[]; out=[]
    for e in ev:
        t=e['type']['name']
        if t=='Shot':
            shots.append((e['possession'],e['team']['id'],e['period'],e['minute']*60+e['second'],e['shot']['outcome']['name'],e['shot'].get('statsbomb_xg',0)))
    for e in ev:
        t=e['type']['name']
        if e['id'] not in ff: continue
        if t=='Shot' and e['shot'].get('type',{}).get('name')=='Open Play':
            pass
        elif t=='Pass':
            p=e['pass']
            if p.get('type',{}).get('name') in ('Corner','Free Kick','Throw-in','Kick Off','Goal Kick'): continue
            if e['location'][0]<80: continue
        else: continue
        if t=='Shot' and e['shot'].get('type',{}).get('name')!='Open Play': continue
        rec=dict(match=mid,comp=comp,id=e['id'],type=t,poss=e['possession'],team=e['team']['id'],period=e['period'],
                 time=e['minute']*60+e['second'],loc=e['location'],ff=[(f['teammate'],f['keeper'],f['actor'],f['location'][0],f['location'][1]) for f in ff[e['id']]])
        if t=='Pass':
            rec['end']=e['pass']['end_location']; rec['pass_ok']=('outcome' not in e['pass'])
        else:
            rec['xg']=e['shot'].get('statsbomb_xg',0); rec['sh_out']=e['shot']['outcome']['name']
        ontarget=0; goal=0
        for (pp,tm,per,tim,o,xg) in shots:
            if pp==rec['poss'] and tm==rec['team'] and per==rec['period'] and rec['time']<=tim<=rec['time']+10 and o in ('Goal','Saved','Saved to Post'): ontarget=1
            if pp==rec['poss'] and tm==rec['team'] and per==rec['period'] and rec['time']<=tim<=rec['time']+10 and o=='Goal': goal=1
        rec['y']=ontarget; rec['y_goal']=goal
        out.append(rec)
    return out
jobs=[]
for cid,sid,name in COMPS:
    ms=get(f'{RAW}/matches/{cid}/{sid}.json') or []
    print(name,len(ms),flush=True)
    jobs+= [(m['match_id'],name) for m in ms]
res=[]
with ThreadPoolExecutor(12) as ex:
    for i,r in enumerate(ex.map(proc,jobs)):
        if r: res+=r
        if i%40==0: print(i,len(res),flush=True)
pickle.dump(res,open('records2.pkl','wb'))
print('done',len(res),sum(1 for r in res if r['type']=='Shot'),sum(1 for r in res if r['type']=='Pass'),len(set(r['match'] for r in res)))
