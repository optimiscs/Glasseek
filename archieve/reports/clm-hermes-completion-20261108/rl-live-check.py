"""Small real-vLLM sampler check, without starting an optimizer or verl cluster."""
import asyncio,json,os,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/root/autodl-tmp/Glasseek');OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'RL_training')]
os.environ['GLASSEEK_CLM_ROLLOUT_ROOT']=str(OUT/'rl-live')
from glasseek_training.official_clm import rollout
from glasseek_training.rollout import RolloutSettings
class Tokenizer:
    """Use the running model's exact tokenizer and chat template."""
    eos_token='<|im_end|>';pad_token='<|endoftext|>'
    def call(self,path,body):
        request=urllib.request.Request('http://127.0.0.1:19109'+path,
            data=json.dumps({'model':'glasseek-base-9b',**body}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=30) as response:return json.load(response)
    def apply_chat_template(self,messages,**kwargs):
        return self.call('/tokenize',dict(messages=messages,tools=kwargs['tools'],
            add_generation_prompt=kwargs['add_generation_prompt'],
            chat_template_kwargs={'enable_thinking':kwargs.get('enable_thinking',True)}))['tokens']
    def decode(self,ids,**kwargs):return self.call('/detokenize',{'tokens':ids})['prompt']
tokenizer=Tokenizer()
recorded=[]
async def generate(prompt,budget):
    body={'model':'glasseek-base-9b','prompt':prompt,'max_tokens':budget,'temperature':1.0,
          'top_p':1.0,'top_k':-1,'logprobs':1,'return_token_ids':True,'seed':42}
    request=urllib.request.Request('http://127.0.0.1:19109/v1/completions',
        data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    def call():
        with urllib.request.urlopen(request,timeout=180) as response:return json.load(response)
    response=await asyncio.to_thread(call)
    choice=response['choices'][0]
    ids=choice['token_ids'];probs=choice['logprobs']['token_logprobs']
    assert len(ids)==len(probs) and all(isinstance(p,(int,float)) for p in probs)
    recorded.append(dict(prompt_ids=list(prompt),response_ids=list(ids),log_probs=list(probs),finish_reason=choice['finish_reason']))
    return ids,probs
class Tools:pass
settings=RolloutSettings(max_turn_tokens=4096,max_turns=6)
started=time.time()
try:
    result=asyncio.run(rollout(tokenizer,generate,Tools(),settings,[{'role':'user','content':
        'First use the bash tool to calculate 17 * 23 with Python. After reading its output, answer directly with Exact Answer: <number>.'}],{'enable_thinking':True}))
    assert result['stop_reason']=='answer' and '391' in result['final_answer']
    assert len(result['segments'])==len(recorded)>=2
    comparisons=[]
    for segment,sampled in zip(result['segments'],recorded):
        comparisons.append(dict(prompt_equal=segment['prompt_ids']==sampled['prompt_ids'],
            token_ids_equal=segment['response_ids']==sampled['response_ids'],
            logprobs_equal=segment['log_probs']==sampled['log_probs'],
            mask_equal=segment['response_mask']==[1]*len(sampled['response_ids']),tokens=len(sampled['response_ids'])))
    assert all(all(v for k,v in c.items() if k!='tokens') for c in comparisons)
    report=dict(status='passed',elapsed_seconds=time.time()-started,answer=result['final_answer'],
        stop_reason=result['stop_reason'],completion_protocol=result['completion_protocol'],
        artifacts=result['official_clm_artifacts'],comparisons=comparisons)
    (OUT/'rl-live-samples.json').write_text(json.dumps(recorded))
except BaseException as exc:
    report=dict(status='failed',elapsed_seconds=time.time()-started,error=repr(exc))
    raise
finally:
    (OUT/'rl-live-check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))
