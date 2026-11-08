import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'RL_training'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'RL_training/tests'))
from test_rollout import Tokenizer
from glasseek_training.rollout import SearchRollout,RolloutSettings
from glasseek_training.web_tools import WebTools

@pytest.mark.parametrize('mode',['raw','hybrid','ref','evidence'])
def test_real_tool_adapter_preserves_every_generated_token_logprob_and_mask(mode):
    url='https://example.org/page?signature=Long%2Fvalue'
    calls=[{'name':'web_search','arguments':{'query':'question'}},
           {'name':'web_extract','arguments':{'url':url} if mode=='raw' else {'urls':['U1']}}]
    outputs=['<tool_call>'+json.dumps(c)+'</tool_call>' for c in calls]+['<answer>Answer</answer>']
    actual=[];sampled=[];tokens=Tokenizer()
    class LocalTools(WebTools):
        def _execute(self,call):
            actual.append(call)
            return json.dumps({'results':[{'url':url,'title':'Page','content':'Evidence.'}]}) if call['name']=='web_search' else 'Evidence.'
    with patch.dict(os.environ,{'GLASSEEK_SEARCH_POOL_ENDPOINT':'http://127.0.0.1:1/search'}):tools=LocalTools(url_mode=mode)
    async def generate(prompt,budget):
        ids=tokens.encode(outputs[len(sampled)]);probs=[-0.123-i/10000 for i in range(len(ids))]
        sampled.append((ids[:],probs[:]))
        return ids,probs
    settings=RolloutSettings(max_model_len=40000,max_prompt_length=12000,max_response_length=20000,compact_at=20000,max_turn_tokens=1000,max_observation_tokens=3000,summary_tokens=500,summary_reserve=1000,max_turns=4)
    result=asyncio.run(SearchRollout(tokens,generate,tools,settings).run([{'role':'user','content':'Question'}]))
    got_ids=[];got_probs=[]
    for segment in result['segments']:
        for token,prob,mask in zip(segment['response_ids'],segment['log_probs'],segment['response_mask']):
            if mask:got_ids.append(token);got_probs.append(prob)
            else:assert prob==0.0
    assert got_ids==[t for ids,probs in sampled for t in ids]
    assert got_probs==[p for ids,probs in sampled for p in probs]
    assert actual==[calls[0],{'name':'web_extract','arguments':{'url':url}}]
    assert result['stop_reason']=='answer'
