import collections,json,os,sqlite3,time
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek/analysis/no-tool-screen')
while True:
 try:
  source=sqlite3.connect(f'file:{P}/results.sqlite3?mode=ro',uri=True,timeout=10);tmp=P/'snapshot.sqlite3.tmp';tmp.unlink(missing_ok=True);target=sqlite3.connect(tmp);source.backup(target);target.close();source.close();os.replace(tmp,P/'snapshot.sqlite3')
  c=sqlite3.connect(P/'snapshot.sqlite3');a=dict(c.execute('select status,count(*) from answers group by status'));j={f'{s}:{l}':n for s,l,n in c.execute('select status,label,count(*) from judgements group by status,label')};c.close();report={'updated_at':time.time(),'unique_inputs_total':2711,'trajectories_total':3265,'inference':a,'judgements':j,'export_finished':Path('/root/autodl-tmp/Glasseek/data/glasseek/sft-no-tool-screened-64k/manifest.json').exists()};(P/'progress.json.tmp').write_text(json.dumps(report,ensure_ascii=False,indent=2));os.replace(P/'progress.json.tmp',P/'progress.json')
  if report['export_finished']:break
 except Exception as e:print(type(e).__name__,str(e)[:200],flush=True)
 time.sleep(30)
