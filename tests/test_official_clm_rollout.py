import asyncio
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'RL_training'))
from glasseek_training.official_clm import rollout
from glasseek_training.rollout import RolloutSettings

class Tokenizer:
    eos_token='<|im_end|>'
    def encode(self,text,**kwargs):return [ord(c) for c in text]
    def decode(self,ids,**kwargs):return ''.join(map(chr,ids))
    def apply_chat_template(self,messages,**kwargs):
        return self.encode(json.dumps(messages,ensure_ascii=False)+'<assistant>')
class Tools:pass

def test_upstream_edits_do_not_rewrite_rl_sampled_actions(tmp_path,monkeypatch):
    monkeypatch.setenv('GLASSEEK_CLM_ROLLOUT_ROOT',str(tmp_path))
    tok=Tokenizer();recorded=[]
    async def generate(prompt,budget):
        n=len(recorded)
        if n==0:command="python3 -c \"print('verified evidence ' * 200)\""
        elif n==1:
            text=tok.decode(prompt);mirror=re.search(r'mirrored to `([^`]+)`',text)[1]
            command="python3 - <<'END'\nfrom pathlib import Path\np=Path("+repr(mirror)+");s=p.read_text();p.write_text(s.split('\\n',1)[0]+'\\nEvidence retained.\\n')\nEND"
        else:
            text='<think>Evidence verified.</think>READY'+tok.eos_token
        if n<2:text='<tool_call>'+json.dumps({'name':'bash','arguments':{'command':command}})+'</tool_call>'
        ids=tok.encode(text);probs=[-(i+1)/10000 for i in range(len(ids))];recorded.append((list(prompt),ids,probs));return ids,probs
    settings=RolloutSettings(max_model_len=100000,max_prompt_length=20000,max_response_length=60000,compact_at=50000,
        max_turn_tokens=4096,max_observation_tokens=8192,summary_tokens=4096,summary_reserve=6144,max_turns=8)
    result=asyncio.run(rollout(tok,generate,Tools(),settings,[{'role':'user','content':'Preserve evidence; answer READY.'}]))
    assert result['final_answer']=='READY' and result['compactions']>=1
    assert len(result['segments'])==len(recorded)==3
    assert result['stop_reason']=='answer' and result['completion_protocol']=='hermes-assistant-v1'
    assert not list(tmp_path.glob('**/answer.txt'))
    for segment,(prompt,ids,probs) in zip(result['segments'],recorded):
        assert segment['prompt_ids']==prompt and segment['response_ids']==ids
        assert segment['log_probs']==probs and segment['response_mask']==[1]*len(ids)
