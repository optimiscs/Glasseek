"""Source-to-output integrity spot check; never replay dataset tools."""
import argparse
import collections
import json
from pathlib import Path
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trajectory_preprocess'))
from ingest import records,normalize
from tool_xml import CALL,THINK

def without_calls(text):
    protected=[(m.start(),m.end()) for m in THINK.finditer(text)]
    parts=[];cursor=0
    for m in CALL.finditer(text):
        if any(a<=m.start()<b for a,b in protected):continue
        parts.append(text[cursor:m.start()]);cursor=m.end()
    parts.append(text[cursor:])
    return re.sub(r'\s+',' ',''.join(parts)).strip()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    picks={};counts=collections.Counter()
    for line in (args.output/'clean.jsonl').open():
        item=json.loads(line);groups=['dataset:'+item['source']]
        groups+=['tool:'+n for n in {c['name'] for s in item['steps'] for c in s['calls']}]
        groups+=['flag:'+f for f in item['harness']['flags']]
        wanted=[g for g in groups if counts[g]<3]
        if wanted:
            picks[item['id']]=item
            counts.update(wanted)
    byfile=collections.defaultdict(dict)
    for item in picks.values():
        p=item['provenance'];byfile[p['path']][p['row_index']]=item
    checked=[]
    for path,wanted in byfile.items():
        for index,raw in enumerate(records(path)):
            if index in wanted:
                item=wanted[index];old=normalize(raw,item['source'],path,index)
                assert old['content_hash']==item['source_content_hash']
                for k in ('question','reference_answer','context_messages','instruction_messages'):
                    assert old.get(k)==item.get(k),(item['id'],k)
                assert len(old['steps'])==len(item['steps'])
                for a,b in zip(old['steps'],item['steps']):
                    assert {k:v for k,v in a.items() if k not in ('calls','assistant')}=={k:v for k,v in b.items() if k not in ('calls','assistant')}
                    assert without_calls(a['assistant'])==without_calls(b['assistant'])
                checked.append({'id':item['id'],'source':item['source'],'provenance':item['provenance'],'status':'preserved',
                                'tool_names':sorted({c['name'] for s in item['steps'] for c in s['calls']})})
            if index>=max(wanted):break
    assert len(checked)==len(picks)
    result={'status':'passed','sample_count':len(checked),'sampling':'First 3 candidates per dataset, tool name and adapter flag; union deduplicated. Integrity check, not a semantic-quality sample.',
            'checks':['source hash','question and reference','all observations','system/developer/user context','non-tool assistant text (ignoring whitespace introduced between split calls)'],
            'samples':checked}
    (args.output/'audit/preservation_checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('Preservation checks passed:',len(checked))
if __name__=='__main__':main()
