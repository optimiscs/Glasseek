"""Official CLM agent loop with verl's existing token/logprob sampler."""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from .web_tools import parse_tool_calls

async def rollout(tokenizer, generate, tools, settings, messages, template_kwargs=None):
    from glasseek_shared.context.official.run import run
    from glasseek_shared.context.official import activate,ROOT,COMMIT
    activate()
    from clm_harness.utils.tool_schemas import BASH_TOOL
    from litellm import ModelResponse
    sys.path.insert(0,str(ROOT/'Evaluation_backend'))
    from agent.session_recorder import AgentSessionRecorder
    segments=[]
    async def query(agent, model, context):
        recorder.before_query(context)
        started=time.time()
        prompt=tokenizer.apply_chat_template(context,tools=[BASH_TOOL],add_generation_prompt=True,
            tokenize=True,**(template_kwargs or {}))
        budget=min(settings.max_turn_tokens,settings.max_model_len-len(prompt))
        if budget<=0:raise ValueError('Official CLM request has no output headroom')
        ids,probs=await generate(list(prompt),budget)
        if not ids or len(ids)!=len(probs) or len(ids)>=budget:
            raise ValueError('Incomplete CLM rollout generation; refusing a partial training sample')
        segments.append(dict(prompt_ids=list(prompt),response_ids=list(ids),response_mask=[1]*len(ids),log_probs=list(probs)))
        text=tokenizer.decode(ids,skip_special_tokens=False)
        calls=parse_tool_calls(text)
        # Interpret the sampled action for execution only. The stored IDs and
        # behavior logprobs above remain exactly those returned by generate().
        parts=text.rsplit('</think>',1)
        reasoning=parts[0].removeprefix('<think>') if len(parts)==2 else None
        content=parts[-1]
        for token in (getattr(tokenizer,'eos_token',None),getattr(tokenizer,'pad_token',None)):
            if token and content.endswith(token):content=content[:-len(token)]
        if calls:content=content.split('<tool_call>',1)[0]
        native=[{'id':uuid4().hex,'type':'function','function':{'name':c['name'],'arguments':json.dumps(c['arguments'])}} for c in calls]
        response=ModelResponse(model='openai/'+model,choices=[{'index':0,'message':{'role':'assistant','content':content,'reasoning_content':reasoning,'tool_calls':native or None},'finish_reason':'tool_calls' if calls else 'stop'}],
            usage={'prompt_tokens':len(prompt),'completion_tokens':len(ids),'total_tokens':len(prompt)+len(ids)})
        recorder.model_response(response,started,time.time())
        return response
    output=Path(os.getenv('GLASSEEK_CLM_ROLLOUT_ROOT',str(ROOT/'Evaluation/data/runtime/clm-rollouts')))/uuid4().hex
    question='\n'.join(str(m.get('content') or '') for m in messages)
    args=SimpleNamespace(question=question,output=output,model='rollout',base_url='',environment='local',host='',image='',
        max_steps=settings.max_turns,max_tokens=settings.max_turn_tokens,context_budget=settings.max_model_len-settings.max_turn_tokens,resume_from=None)
    recorder=AgentSessionRecorder(output/'hermes',output/'sessions/session_rollout.json','rollout-'+output.name,'rollout',cwd=str(output/'workspace'),tools=[BASH_TOOL],capacity_tokens=settings.max_model_len)
    try:
        result=await run(args,query_hook=query,tools=tools,recorder=recorder)
        recorder.finish()
    except BaseException:
        recorder.finish('interrupted')
        raise
    finally:recorder.close()
    if not segments:raise RuntimeError('CLM produced no trainable actions')
    return {'segments':segments,'final_answer':result['answer'],'stop_reason':result['stop_reason'],
            'num_turns':len(segments),'compactions':result['usage']['n_ctx_syncs_real'],
            'context_edits':[],'completion_protocol':result['protocol'],'official_clm_commit':COMMIT,'official_clm_artifacts':str(output)}
