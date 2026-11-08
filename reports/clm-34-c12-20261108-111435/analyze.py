import json
from pathlib import Path
from collections import Counter
OUT=Path(__file__).resolve().parent;ROOT=Path('/root/autodl-tmp/Glasseek');state=json.loads((OUT/'status.json').read_text());run=state['runs']['clm']['run'];folder=ROOT/'Evaluation/output/preds'/run/state['dataset'];rows=[]
import subprocess
subprocess.run([str(ROOT/'.cache/clm-official-env/bin/python'),str(ROOT/'scripts/trajectory_benchmark/migrate_clm_sessions.py'),'--run',run,'--dataset',state['dataset'],'--report',str(OUT/'native-recording-migration.json')],check=True)
for path in (folder/'conv').rglob('*.json'):
 c=json.loads(path.read_text());j=folder/'judge_official'/c['question_id']/(c['run_id']+'.json');judge=json.loads(j.read_text()) if j.exists() else {}
 sidecar=Path(c['workspace_dir'])/'official-clm-result.json';extra=json.loads(sidecar.read_text()) if sidecar.exists() else {};u=extra.get('usage',{})
 rows.append({'question_id':c['question_id'],'status':c['status'],'score':judge.get('score'),'judge_status':judge.get('status'),'answer':c.get('model_response'),'tool_calls':c.get('tool_calls',{}),'wall_seconds':c.get('wall_seconds'),'input_tokens':u.get('prompt_tokens'),'output_tokens':u.get('completion_tokens'),'cached_input_tokens':max(u.get('cached_tokens',0),u.get('cache_read_input_tokens',0)),'context_edits':u.get('n_ctx_syncs_real',0),'rollbacks':u.get('n_retry_on_limit',0)})
counts=Counter()
for r in rows:counts.update(r['tool_calls'])
report={'run':run,'n':len(rows),'correct':sum(r['score']==1 for r in rows),'accuracy_over_34':sum(r['score']==1 for r in rows)/34,'counts':dict(counts),'rows':rows,'source_drift':state.get('source_drift',[]),'token_semantics':'Official usage.prompt_tokens includes cache reads; do not add cached_input_tokens again.','limits':'Official CLM changes both prompt/tool protocol and context management; concurrency differs from previous URL-mode runs, so comparisons are descriptive.'}
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
