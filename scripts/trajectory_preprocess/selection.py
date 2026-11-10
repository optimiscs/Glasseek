"""Hard gates precede diversity ordering; no score can override a hold."""
from client import digest
from collections import defaultdict,deque


def diverse_order(rows):
    """Round-robin eligible strata, keeping families together in one split bucket."""
    groups=defaultdict(list)
    for row in rows:
        if row.get('selection',{}).get('eligible_for_ranking'):
            groups[tuple(row['selection']['stratum'])].append(row)
    queues=[deque(sorted(v,key=lambda r:(len(r.get('annotation_issues',[])),r['id']))) for k,v in sorted(groups.items())]
    ordered=[]
    while any(queues):
        for q in queues:
            if q: ordered.append(q.popleft()['id'])
    return ordered


def describe(item, canonical, result):
    routing=result['eligibility']['routing']
    inventory=result.get('inventory',{})
    length=sum(len(s['assistant'])+len(s['observation']) for s in item['steps'])
    language='zh' if sum('\u4e00'<=c<='\u9fff' for c in item['question'])>len(item['question'])/5 else 'other'
    family=canonical.get('split_group_id') or canonical.get('bundle_id') or item.get('question_hash') or digest(item['question'])
    return {'group_id':family,'split_bucket':int(digest(family)[:8],16)%100,
            'stratum':[item['source'],inventory.get('task_goal',routing['task_goal']),routing['agent_topology'],language,
                       'short' if length<20000 else 'medium' if length<80000 else 'long'],
            'difficulty':inventory.get('difficulty','unknown'),'topic':inventory.get('topic','unknown'),
            'semantic_labels_status':'unverified_model_labels', 'strategy':[s['type'] for s in result.get('segments',[])],
            'quality_vector':{'evidence_gaps':len(result.get('evidence_gaps',[])),
                              'quality_issues':len(result.get('quality_issues',[])),
                              'annotation_issues':len(result.get('annotation_issues',[]))},
            'eligible_for_ranking':bool(result.get('offline_quality_candidate') and result.get('automatic_acceptance_passed') and
                result.get('supervision') and not any(r['decision']=='review' for r in result['supervision']['messages']) and
                any(r['decision']=='learn' for r in result['supervision']['messages'])),
            'selection_status':'human_review_pending','training_ready':False}
