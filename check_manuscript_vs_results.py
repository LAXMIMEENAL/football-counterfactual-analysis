"""Checks that key numbers printed in the revised manuscript (docx) match the result files. Usage: python check_manuscript_vs_results.py manuscript.docx"""
import json,re,sys,zipfile
x=zipfile.ZipFile(sys.argv[1]).read('word/document.xml').decode('utf8')
t=re.sub(r'<[^>]+>','',x).replace('&amp;','&')
a=json.load(open('results5a.json')); b=json.load(open('results5b.json')); r2=json.load(open('results2.json'))
mn=lambda s:str(s).replace('-','−'); pct=lambda v:f"{100*v:.1f}%"
exp=[f"{a['overlap']['domain_auc']:.2f}",pct(b['main_4m']['feas1']),pct(b['main_4m']['lane1']),pct(b['main_4m']['near1']),f"{b['completion_model']['auc']:.3f}",
     mn(f"{a['ablation_goal']['Defensive Context'][0]:+.3f} [{a['ablation_goal']['Defensive Context'][1]:+.3f}, {a['ablation_goal']['Defensive Context'][2]:+.3f}]"),
     pct(b['cds_feasibility']['feas_pos']),pct(a['cds_overlap']['all']['pos'])]
for k,v in b['threshold'].items(): exp+= [pct(v['lane1']),pct(v['near1'])]
for k in ('shots','passes'): exp+=[f"{a['calibration'][k]['brier']:.4f}",f"{a['calibration'][k]['slope']:.2f}"]
miss=[e for e in exp if e not in t]; print('checked',len(exp),'strings in manuscript; missing:',miss)
