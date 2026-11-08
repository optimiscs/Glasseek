import json, multiprocessing as mp
from pathlib import Path
from glasseek_training import filter_sft_length as f

ROOT=Path('/root/autodl-tmp/Glasseek')

def count(item):
    line_no, raw, expected = item
    row=json.loads(raw)
    enabled=bool(row.get('enable_thinking','none'))
    total=supervised=0
    for idx,message in enumerate(row['messages']):
        ids,mask,_,_=f._dataset._process_single_message(idx,message,row['messages'],row.get('tools') if idx==0 else None,enabled)
        total+=len(ids)
        supervised+=int(mask.sum())
    if total != expected:
        raise ValueError(f'line {line_no}: expected {expected} tokens, got {total}')
    return {'source_line':line_no,'tokens':total,'supervised_tokens':supervised}

def main():
    old=[json.loads(x) for x in (ROOT/'data/glasseek/sft/selection.jsonl').open() if json.loads(x)['exact']]
    extra=[json.loads(x) for x in (ROOT/'reports/experiments/sp4-30g/lengths-extra.jsonl').open()]
    selection={x['source_line']:x['tokens'] for x in old+extra if x['tokens']<=131072}
    assert len(selection)==5749
    def items():
        for i,line in enumerate((ROOT/'data/Glasseek-SFT-5.7K/train.jsonl').open(),1):
            if i in selection:yield (i,line,selection[i])
    out=ROOT/'reports/experiments/pro6000-lora/supervised-counts.jsonl'
    best=None
    with out.open('w') as target, mp.Pool(8,initializer=f.initialize,initargs=('/root/autodl-tmp/models/Qwen3.5-9B',)) as pool:
        for n,result in enumerate(pool.imap_unordered(count,items(),chunksize=2),1):
            target.write(json.dumps(result)+'\n')
            if best is None or result['supervised_tokens']>best['supervised_tokens']:best=result
            if n%500==0:
                target.flush();print('counted',n,'of',len(selection),'best',best,flush=True)
    print('FINAL_BEST',json.dumps(best),flush=True)
if __name__=='__main__':main()
