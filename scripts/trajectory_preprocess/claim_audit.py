"""Program-owned claim/edge coverage with unchanged evidence validators."""
import json
import re
import policy
from glasseek_shared.curation.schema import obj, enum, array, TEXT, BOOL, STATUSES


def structured(client, stage, ident, prompt, schema, validator):
    prompt += '\nWIRE FORMAT OVERRIDE: use exactly the attached JSON Schema, not the illustrative JSON above. '
    prompt += 'Claim, evidence-edge, step and segment ID keys supplied by the program are immutable; assess each independently. All schema-required fields must be present.\n' + json.dumps(schema)
    return client.call(stage,ident,prompt,policy.SYSTEM,validator,schema=schema)


def inventory(item, client):
    sources = policy.inventory_sources(item)
    def entry(kind):
        return obj({'text':TEXT,'source_span_id':enum(r['span_id'] for r in sources[kind])})
    fmt = obj({**entry('question')['properties'], 'type':enum(sorted(policy.FORMAT_TYPES)),
               'requirement_kind':enum(['answer_type','presentation'])})
    schema = obj({'constraints':array(entry('question')),'task_requirements':array(entry('question')),'claims':array(entry('answer'),1),
                  'format_requirements':array(fmt),
                  'task_goal':enum(['factual_qa','open_analysis','calculation','mixed','unknown']),
                  'topic':TEXT,'difficulty':enum(['low','medium','high','unknown'])})
    def validate(value):
        for row in value['constraints']:
            if re.match(r'(?i)^(?:identify|find|determine|provide|return|give)\s+(?:the|this|that|which|what|an?|only)\b|^请(?:识别|找出|回答|输出|确定)',row['text'].strip()):
                raise ValueError('Task instruction in factual constraints; keep the instruction under task_requirements and express factual relations separately')
        for kind,prefix in [('constraints','q'),('task_requirements','r'),('claims','a'),('format_requirements','f')]:
            for i,row in enumerate(value[kind],1): row['id'] = prefix+str(i)
        return policy.validate_inventory(value,item)
    prompt=policy.inventory_prompt(item)+'\nSeparate task_requirements (what the user asks the assistant to do) from constraints (world-factual relations or clues that evidence must establish). "Identify X" is an instruction, not a world-factual claim. Do not require web evidence proving the instruction was issued. Keep every instruction under task_requirements, even when constraints is empty. Requirements are still checked by the independent full-trajectory requirements_satisfied gate. Express factual clues as predicates, not imperative commands. Also classify task_goal, topic and difficulty for stratified review; these are tentative semantic labels, never acceptance criteria. For open analysis, distinguish factual premises from recommendations; do not invent factual requirements to force a verdict.'
    return structured(client,'inventory',item['id'],prompt,schema,validate)


def audit(item, inv, client):
    targets = [r['id'] for k in ('constraints','claims') for r in inv[k]]
    spans = [r['span_id'] for r in policy.evidence_catalog(item)]
    proof = obj({'op':enum(['subtract','add','calendar_interval']), 'left':{'type':'integer'},
                 'right':{'type':'integer'},'result':{'type':['integer','boolean']},
                 'partition_count':{'type':['integer','null']},'partition_index':{'type':['integer','null']}})
    def judgment(statuses, minimum):
        return obj({'status':enum(statuses),'reason':TEXT,'evidence':array(obj({'span_id':enum(spans)}),minimum),
                    'derivation':{'anyOf':[{'type':'null'},proof]}})
    entry = {'anyOf':[judgment(['insufficient'],0),judgment(['supported','contradicted','conflict'],1)]}
    schema = obj({ident:entry for ident in targets})
    def validate(value):
        rows=[]
        for ident in targets:
            row={'id':ident,**value[ident]}
            if row['derivation']:
                row['derivation']={k:v for k,v in row['derivation'].items() if v is not None}
            rows.append(row)
        result=policy.validate_audit({'items':rows,'final_matches_reference':None},item,inv)
        for row in result['items']:
            if row['status']!='insufficient' and not row['evidence']:
                raise ValueError('Initial '+row['status']+' verdict for '+row['id']+' has no located evidence; cite admissible evidence or return insufficient. This is an annotation error, not proof the source is invalid.')
        return result
    return structured(client,'audit',item['id'],policy.audit_prompt(item,inv),schema,validate)


def review(item, inv, audited, client):
    edge = obj({'relation':enum(['supports','contradicts','context']),'reason':TEXT,
                'same_scope':BOOL,'scope_reason':TEXT})
    schema = obj({r['id']:obj({'verdict':STATUSES,'requires_computation':BOOL,'derivation_relevant':BOOL,
                 'reason':TEXT,'edges':obj({e['evidence_id']:edge for e in r['evidence']})}) for r in audited['items']})
    def validate(value):
        rows=[]
        for ident,row in value.items():
            rows.append({'id':ident,**row,'edges':[{'evidence_id':eid,**e} for eid,e in row['edges'].items()]})
        return policy.validate_review({'items':rows},audited)
    return structured(client,'review',item['id'],policy.review_prompt(inv,audited),schema,validate)


def behaviors(item, inv, reviewed, client):
    evidence = [r['span_id'] for r in policy.evidence_catalog(item)]
    def events(step):
        assistant=[r['span_id'] for r in policy.spans(step['assistant'],f'B{step["step_id"]}')]
        def variant(tags, spans):
            return obj({'tag':enum(tags),'target_id':TEXT,'reason':TEXT,'span_id':enum(spans)})
        return array({'anyOf':[
            variant(['reasonable_exploration','repeat','tool_failure'],['']+evidence+assistant),
            variant(['discover_supported','verify_supported','exclude_wrong'],evidence),
            variant(['reject_supported','unsupported_assertion','ignore_conflict'],assistant)]})
    schema = obj({str(s['step_id']):events(s) for s in item['steps']})
    def validate(value):
        return policy.validate_behaviors({'steps':[{'step_id':s['step_id'],'events':value[str(s['step_id'])]} for s in item['steps']]},item,reviewed)
    return structured(client,'behaviors',item['id'],policy.behavior_prompt(item,inv,reviewed),schema,validate)
