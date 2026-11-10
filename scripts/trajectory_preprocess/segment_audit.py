"""Semantic segments; ex-ante requests cannot see their own or future returns."""
import json
import policy
from claim_audit import structured
from glasseek_shared.curation.schema import obj, enum, TEXT, BOOL
from glasseek_shared.curation.supervision import build_plan

TYPES = ['task_decomposition','candidate_search','targeted_verification','conflict_resolution','failure_recovery','answer_synthesis']


def visible_step(step):
    # The normalized record also contains the same returns in `responses` plus
    # migration metadata. Send each original model-visible text exactly once.
    return {k:step[k] for k in ('step_id','assistant','calls','observation')}


def causal_prefix(item, step_id):
    # Later instructions and same-step observations are not available at action time.
    contexts = [c for c in item.get('context_messages',[])+item.get('instruction_messages',[]) if c['before_step'] <= step_id]
    return {'question':item['question'],'context_messages':contexts,
            'steps':[{k:s[k] for k in ('step_id','assistant','calls')} if s['step_id']==step_id else visible_step(s)
                     for s in item['steps'] if s['step_id']<=step_id]}


def partition(item, client):
    ids = [s['step_id'] for s in item['steps']]
    # Every assistant boundary is a permitted cut; no XML/tool envelope is split.
    schema = obj({str(i):obj({'start_new_segment':{'const':True,'type':'boolean'} if i==ids[0] else BOOL,
                             'type':enum(TYPES),'purpose':TEXT}) for i in ids})
    prompt = ('Merge consecutive assistant steps into semantic search-purpose segments. Cover every step once, in original order. '
              'A search followed by conflict checking and correction can be one segment. Preserve exploration and recovery. '
              'For EVERY fixed step key, choose start_new_segment=true to begin a new purpose, or false to merge into the preceding segment. '
              'The first step must start a segment. Type and purpose of each segment come from its starting step; continuation labels are descriptive only. '
              'Do not assess earlier reasonableness using future outcomes. This request only proposes boundaries. DATA:\n'+
              json.dumps({'question':policy.task_text(item),'steps':[visible_step(s) for s in item['steps']]},ensure_ascii=False))
    def validate(value):
        if set(value)!={str(i) for i in ids} or value[str(ids[0])]['start_new_segment'] is not True:
            raise ValueError('Incomplete fixed-step boundary decisions')
        segments=[]
        for i in ids:
            row=value[str(i)]
            if row['start_new_segment']:
                segments.append({'id':f'segment_{len(segments)+1}','start_step':i,'end_step':i,
                                 'type':row['type'],'purpose':row['purpose']})
            else:segments[-1]['end_step']=i
        return segments
    return structured(client,'segment_partition',item['id'],prompt,schema,validate)


def evaluate(item, canonical, result, client):
    segments = partition(item,client)
    # Evaluate each action with its own prefix. Even a segment's later actions
    # must not leak their observations into evaluation of its first action.
    schema = obj({'verdict':enum(['reasonable','unreasonable','uncertain']),'reason':TEXT})
    actions={}
    for step in item['steps']:
        ident=step['step_id']
        prompt=('Judge only the last assistant action using information available BEFORE its tool return. '
                'Reasonable exploration, candidate exclusion, verification and failure recovery may be valuable even if unsuccessful. '
                'Do not guess unseen observations. A repeated query is not automatically unreasonable. DATA:\n'+
                json.dumps(causal_prefix(item,ident),ensure_ascii=False))
        actions[ident]=structured(client,'segment_causal',item['id']+':step:'+str(ident),prompt,schema,lambda v:v)
    schema = obj({s['id']:obj({'contribution':enum(['progress','reasonable_exploration','redundant','harmful','uncertain']),
                              'reason':TEXT}) for s in segments})
    reviewed=[{'id':r['id'],'status':r['status'],'reason':r['reason'],
               'evidence':[{'span_id':e['span_id'],'step_id':e['step_id']} for e in r['evidence']],
               'checked_derivation':r['checked_derivation']} for r in result['audit']['items']]
    prompt=('Assess ex-post contribution of each semantic segment using the complete recorded trajectory and reviewed evidence. '
            'Useful exclusion and failed exploration are valid contributions. Do not reinterpret ex-ante reasonableness here. '
            'A shorter trajectory or more URLs is not intrinsically better. Never use outside knowledge. DATA:\n'+
            json.dumps({'question':policy.task_text(item),'steps':[visible_step(s) for s in item['steps']],
                        'segments':segments,'audit':reviewed},ensure_ascii=False))
    contributions=structured(client,'segment_contribution',item['id'],prompt,schema,lambda v:v)
    source_indices={s['step_id']:s.get('source_message_index') for s in item['steps']}
    decisions={}
    for segment in segments:
        step_ids=list(range(segment['start_step'],segment['end_step']+1))
        segment['causal_assessments']={str(i):actions[i] for i in step_ids}
        segment['ex_post']=contributions[segment['id']]
        uncertain=any(actions[i]['verdict']=='uncertain' for i in step_ids) or segment['ex_post']['contribution']=='uncertain'
        learn=all(actions[i]['verdict']=='reasonable' for i in step_ids) and segment['ex_post']['contribution'] in ('progress','reasonable_exploration')
        decision='review' if uncertain else 'learn' if learn else 'context_only'
        # Whole-trajectory hard gates dominate all local segment benefits.
        if not result['offline_quality_candidate']: decision='review'
        segment['decision']=decision
        indices=[i for i,m in enumerate(canonical['messages']) if m['role']=='assistant' and m.get('source_message_index') in {source_indices[s] for s in step_ids}]
        if len(indices)!=len(step_ids): raise ValueError('Cannot uniquely map segment to canonical assistant messages')
        segment['message_indices']=indices
        for i in indices: decisions[i]={'decision':decision,'segment_id':segment['id']}
    return segments,build_plan(canonical['messages'],decisions)
