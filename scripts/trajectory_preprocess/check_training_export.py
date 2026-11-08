"""CPU-only dry run through the existing SFT loader and actual target tokenizer."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'RL_training'))


def main():
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf
    from glasseek_training.sft_dataset import GlasseekSFTDataset
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--tokenizer',default='/root/autodl-tmp/models/Qwen3.5-9B')
    args=ap.parse_args()
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    config=OmegaConf.create({'messages_key':'messages','max_length':131072,'pad_mode':'no_padding',
                            'truncation':'error','ignore_input_ids_mismatch':True})
    data=GlasseekSFTDataset(str(args.input),tokenizer,config)
    rows=[]
    for i in range(len(data)):
        example=data[i];masks=[]
        for j,message in enumerate(data.messages[i]):
            ids,mask,attention,extra=data._process_single_message(j,message,data.messages[i],data.tools[i] if j==0 else None,True)
            loss_tokens=int(mask.sum())
            if message['role']!='assistant':assert loss_tokens==0
            elif message['content'].strip():assert loss_tokens>0
            masks.append({'role':message['role'],'tokens':len(ids),'loss_tokens':loss_tokens})
        rows.append({'id':data.dataframe.iloc[i]['id'],'tokens':len(example['input_ids']),
                     'loss_tokens':int(example['loss_mask'].sum()),'messages':masks,
                     'under_64k':len(example['input_ids'])<=65536,'under_128k':len(example['input_ids'])<=131072})
    result={'loader':'GlasseekSFTDataset','tokenizer':args.tokenizer,'rows':rows,
            'non_assistant_loss_zero':True,'training_ready':False,
            'note':'Compatibility dry run only. No model training or human acceptance.'}
    (args.input.parent/'loader_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({**result,'rows':[{k:v for k,v in r.items() if k!='messages'} for r in rows]},ensure_ascii=False))


if __name__=='__main__':main()
