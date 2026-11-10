"""Exercise the actual Glasseek SFT loader's message tokenization and loss masks."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RL_training'))
import torch
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from glasseek_training.sft_dataset import GlasseekSFTDataset
from verl.utils.tokenizer.chat_template import extract_system_prompt_and_generation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--comparison',type=Path,required=True);args=ap.parse_args()
    model='/root/autodl-tmp/models/Qwen3.5-9B';tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
    ds=object.__new__(GlasseekSFTDataset);ds.processor=None;ds.tokenizer=tok;ds.apply_chat_template_kwargs={}
    ds.system_prompt,ds.generation_prompt=extract_system_prompt_and_generation(tok)
    results=[]
    for row in pq.read_table(args.comparison/'mask_probe.parquet').to_pylist():
        messages=json.loads(row['messages']);tools=json.loads(row['tools']);counts=collections.Counter();issues=[]
        for i,m in enumerate(messages):
            ids,mask,attn,_=ds._process_single_message(i,m,messages,tools=tools,enable_thinking=True)
            counts['tokens']+=len(ids);counts['supervised_tokens']+=int(mask.sum());counts['messages']+=1
            if m['role']!='assistant':assert int(mask.sum())==0
            else:
                assert int(mask.sum())>0
                decoded=tok.decode(ids.tolist(),skip_special_tokens=False)
                supervised=tok.decode(ids[mask.bool()].tolist(),skip_special_tokens=False)
                if '<think>' in m['content']:
                    # Exact body check, not just presence of an empty template think block.
                    import re
                    for block in re.findall(r'<think>(.*?)</think>',m['content'],re.S):
                        if block.strip():
                            assert block.strip() in decoded,(row['id'],i,'thinking lost')
                            assert block.strip() in supervised,(row['id'],i,'thinking loss masked unexpectedly')
                            counts['thinking_blocks_preserved']+=1
                if '<tool_call>' in m['content']:assert '<tool_call>' in decoded;counts['tool_call_messages']+=1
            if '<tool_response>' in m['content']:
                assert int(mask.sum())==0;counts['masked_tool_response_messages']+=1
        results.append({'id':row['id'],'pair':row['pair'],'source':row['source'],'agent_role':row['agent_role'],
                        **counts,'fits_128k':counts['tokens']<=131072,'status':'passed'})
    output={'status':'passed','tokenizer':model,'loader':'GlasseekSFTDataset._process_single_message','records':results,
            'scope':'Actual message-token loss masks and preservation checked for 8 parent trajectories and all compatible children of the four selected SearchSwarm calls. Held children are listed in the comparison, not passed as compatibility examples. No model execution or optimizer step. Full corpus lengths not computed.'}
    (args.comparison/'mask_checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');print(json.dumps(output,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
