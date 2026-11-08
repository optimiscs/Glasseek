"""Persist source-based assistant spot checks; these are not human approvals."""
import json,re
from pathlib import Path
OUT=Path('/root/autodl-tmp/Glasseek/reports/trajectory-stratified-qwen38-20261107')
CHECKS=[
 {'id':'b552ad5eaba6a63175af','classification':'annotation_false_hold_risk','target_ids':['q2','q3'],
  'finding':'The full observation supports manufacturer, cartridge and rifle name. The auditor removed Markdown link syntax and changed curly apostrophes in purported exact quotes; validation removed these quotes before blind review. The reviewer then received no passages. This does not establish poor trajectory quality.',
  'locators':[(22,'Vigilance Rifles, based in Valley, Arizona'),(22,'[They produce the M18]')]},
 {'id':'1a5e7a7f8d5d4f97dd73','classification':'annotation_false_hold_risk','target_ids':['q3'],
  'finding':'The actor relation is explicit in the complete cast list and tool summary. The auditor selected only the beginning of the cast list plus a different actor’s filmography for q3, so the blind reviewer correctly rejected those selected passages but the resulting q3 hold is caused by poor evidence selection. This check confirms the actor/role relation, not the unsupported spelling of the Chinese transliteration.',
  'locators':[(3,"[Matt O'Leary]"),(3,'The actor who played the character Eric')]},
 {'id':'b86ca578e6274d18ebfc','classification':'evidence_gap_and_first_stage_overacceptance','target_ids':['q7','q11'],
  'finding':'The first auditor accepted the naming-authority constraint while admitting its passage did not name that authority. Blind review changed it to insufficient. The observation confirms the designation but does not establish the named authority; this is an unresolved support gap, not a proven false answer. The painting-like description also was not established by the selected evidence.',
  'locators':[(2,'川陕革命根据地纪念馆近日在陕南首家揭牌')]},
 {'id':'f151e838ea50f551ba4e','classification':'recorded_evidence_absent','target_ids':[],
  'finding':'This sampled trajectory has one assistant step, zero tool calls and no observation text. The offline policy cannot verify its factual answer against recorded tool evidence. Holding it is justified for this tool-grounded curation task; factual falsity is not established.',
  'locators':[]},
 {'id':'933a7d694655a9d7db6e','classification':'unresolved_constraint_support','target_ids':['q6'],
  'finding':'The interception answer has a direct observation. The selected evidence for the coach’s final game establishes his final season/retirement but does not explicitly establish that game was the final game. No external sports knowledge was used to bridge that gap.',
  'locators':[(20,'Bobby Downs intercepted'),(12,'He retired from coaching after')]},
 {'id':'1dbabcd20f6479a12310','classification':'answer_support_confirmed_in_observations','target_ids':['q1','a1'],
  'finding':'Recorded election coverage explicitly links the Sadrist Movement to the largest party result in October 2021. The final year is supported. This check does not certify every search step as efficient or independently verify the original websites.',
  'locators':[(12,'largest political party in the October 2021')]},
]
CHECKS.extend([
 {'id':'149c5345889417243322','classification':'false_conflict_label_from_lost_scope','target_ids':['q5','a7'],
  'finding':'The blind reviewer promoted insufficient support to a direct conflict. Full step 13 explicitly describes healthy humans at 145 mg/day; step 19 describes Balb/c mice at 100 mg/kg. These are different experimental settings. A lack of evidence of dose dependence in one setting does not directly contradict a generic or existential claim about other settings. The dose-dependent assertion still lacks adequate support, but the conflict label is not established by these passages. No external biomedical facts were used.',
  'locators':[(13,'healthy humans'),(13,'145 mg/day'),(19,'In Balb/c mice'),(19,'100 mg/kg')]},
 {'id':'318da21127373e186f6d','classification':'genuine_tool_observation_conflict_not_proven_answer_error','target_ids':['a10'],
  'finding':'The saved extraction summary says 30 November 2008, while search excerpts from the same UN resolution and UN news say 31 December 2008. The observation conflict is real. The final answer uses the latter date and has supporting excerpts. Holding for source-fidelity review is defensible; this alone does not prove the final answer is wrong. The extraction summary should not be treated as indistinguishable from an original document passage.',
  'locators':[(6,'2008年11月30日'),(7,'2008 年12 月31 日'),(12,'2008年12月31日')]},
 {'id':'222dd87f97f50a619ca2','classification':'supported_constraint_contradiction','target_ids':['q6'],
  'finding':'The question requires a county border to the west of the selected city. The recorded observation places the Taiwan Strait to its west and the counties to its north/northeast and south. The final answer simply identifies Hsinchu City without resolving this discrepancy. The hold is justified by an explicit mismatch within the supplied record, independent of reference-answer agreement or outside geographic knowledge.',
  'locators':[(30,'西临[台湾海峡]'),(30,'南邻[苗栗县]')]},
])

def main():
    artifacts={json.loads(p.read_text())['sample']['id']:json.loads(p.read_text()) for p in (OUT/'results').glob('*.json')}
    output=[]
    for check in CHECKS:
        item=json.loads((OUT/'normalized'/(check['id']+'.json')).read_text());evidence=[]
        for sid,needle in check['locators']:
            observation=item['steps'][sid-1]['observation'];pos=observation.find(needle)
            assert pos>=0,(check['id'],sid,needle)
            evidence.append({'step_id':sid,'raw_offset':pos,'needle':needle,'context':observation[max(0,pos-80):pos+len(needle)+260]})
        result=artifacts.get(check['id'],{}).get('result',{})
        output.append({**{k:v for k,v in check.items() if k!='locators'},'source':item['source'],'evidence':evidence,
                       'observation_characters':sum(len(s['observation']) for s in item['steps']),'tool_calls':sum(len(s['calls']) for s in item['steps']),
                       'pipeline_status':result.get('status','pending'),'pipeline_candidate':result.get('offline_quality_candidate'),
                       'pipeline_blockers':result.get('blockers',[]),'human_approval':False})
    (OUT/'spotcheck.json').write_text(json.dumps({'reviewer_type':'assistant source-based spot check, not independent human gold labels','population_false_positive_rate':'not_estimable','population_false_negative_rate':'not_estimable','cases':output},ensure_ascii=False,indent=2)+'\n')
    print('saved',len(output),'source-based checks')
if __name__=='__main__':main()
