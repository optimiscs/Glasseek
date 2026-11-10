#!/usr/bin/env python3
"""Audit scorer assets/capability gaps without inference, API calls or score substitution."""
import argparse
import importlib.util
import json
from pathlib import Path

from suite import ROOT,dump,read,verify


def probe_vision(endpoint, model, api_key):
    """Exercise image input, not just /models; fail before dispatching tasks."""
    import base64,io
    import httpx
    from PIL import Image
    image=io.BytesIO();Image.new('RGB',(32,32),(255,0,0)).save(image,format='PNG')
    payload={'model':model,'messages':[{'role':'user','content':[
        {'type':'text','text':'Name the single color of this image. Answer one word.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image.getvalue()).decode()}}]}],
        'max_tokens':64,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
    with httpx.Client(timeout=90,trust_env=False) as client:
        r=client.post(endpoint.rstrip('/')+'/chat/completions',json=payload,
                      headers={'Authorization':'Bearer '+api_key})
        r.raise_for_status();response=r.json()
    text=response['choices'][0]['message'].get('content','').strip().lower()
    if text.strip(' .!"。') not in {'red','红','红色'}:raise ValueError('vision smoke did not identify the diagnostic image')
    return {'endpoint':endpoint,'model':model,'status':'verified','response':text}


def audit(suite,corpus_config=None):
    suite=Path(suite).resolve();verify(suite);base=suite.parent
    refs=list(read(suite/'private/references.jsonl'));result={}
    for entry in json.loads((suite/'summary.json').read_text()):
        name=entry['benchmark'];selected=[r for r in refs if r['benchmark']==name];gaps=[]
        if name=='browsecomp_plus':
            if not corpus_config:gaps.append('fixed corpus search/extract routing and retrieved docid event capture must be explicitly configured; live web is not equivalent')
            else:
                cfg=json.loads(Path(corpus_config).read_text());index=Path(cfg['index'])
                if not cfg.get('live_web_disabled') or not index.is_file():gaps.append('fixed corpus index is not ready')
                elif json.loads(index.with_suffix('.manifest.json').read_text()).get('revision')!=cfg['corpus_revision']:gaps.append('corpus revision mismatch')
        if name=='liveresearchbench':
            missing=[r['id'] for r in selected if not (base/'evaluators/liveresearchbench/data/reference_reports'/('qid_'+r['id']+'_report.md')).is_file()]
            if missing:gaps.append({'missing_official_depth_reference_reports':missing,'policy':'retain sampled tasks; depth unscored until author references supplied; other four criteria can be scored separately'})
        if name=='mind2web2':
            missing=[r['id'] for r in selected if not (base/'sources/Mind2Web-2/evaluation_scripts/2025_10_23'/(r['id']+'.py')).is_file()]
            if missing:gaps.append({'missing_native_task_scripts':missing})
        if name=='widesearch':
            missing=[r['id'] for r in selected if not (base/'sources/WideSearch/widesearch_gold'/(r['id']+'.csv')).is_file()]
            if missing:gaps.append({'missing_gold_tables':missing})
        result[name]={'selected':len(selected),'protocol':entry['protocol'],'asset_gaps':gaps,
                      'runtime_requirements':'see setup_score.sh; API/browser/index availability checked at launch, no live judge called in this audit'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--suite',type=Path,default=ROOT/'data/evaluation/suite5pct-v1/release');p.add_argument('--out',type=Path);p.add_argument('--corpus-config',type=Path)
    a=p.parse_args();result=audit(a.suite,a.corpus_config)
    if a.out:dump(a.out,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
