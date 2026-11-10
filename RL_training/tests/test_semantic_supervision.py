"""Real tokenizer + publisher + verl loader, using a clearly synthetic fixture."""
import os
from pathlib import Path
import subprocess
import textwrap
import pytest


def test_real_qwen_token_masks_preserve_context(tmp_path):
    root=Path(__file__).resolve().parents[2]
    python=Path('/root/autodl-tmp/glasseek-verl091-env/bin/python')
    model=Path('/root/autodl-fs/models/Qwen3.5-9B')
    if not python.exists() or not model.exists():
        pytest.skip('Local Qwen tokenizer and pinned verl integration environment required')
    script=textwrap.dedent('''
        import json,sys
        from pathlib import Path
        import torch
        from transformers import AutoTokenizer
        from omegaconf import OmegaConf
        from glasseek_shared.curation.supervision import build_plan
        from glasseek_shared.trajectory_release import payload,digest,runtime_schemas
        from publish_release import publish
        from glasseek_training.sft_dataset import GlasseekSFTDataset
        out=Path(sys.argv[1])
        messages=[{'role':'system','content':'Synthetic tokenizer integration fixture.'},
                  {'role':'user','content':'Compute 1+1.'},
                  {'role':'assistant','content':'This intermediate message stays in context.'},
                  {'role':'user','content':'Give the result.'},
                  {'role':'assistant','content':'<answer>2</answer>'}]
        plan=build_plan(messages,{2:{'decision':'context_only','segment_id':'s1'},
                                  4:{'decision':'learn','segment_id':'s2'}})
        candidate={'id':'synthetic-mask-fixture','source':'synthetic-test','messages':messages,
                   'tools':[{'type':'function','function':v} for v in runtime_schemas().values()],
                   'supervision':plan,'enable_thinking':False}
        (out/'input.jsonl').write_text(json.dumps(candidate)+'\\n')
        # These receipts exist only inside the isolated test fixture directory.
        receipt={'id':candidate['id'],'content_sha256':digest(payload(candidate)),
                 'safety':'passed','quality':'passed','replay':'passed','review':'passed'}
        (out/'receipt.jsonl').write_text(json.dumps(receipt)+'\\n')
        manifest=publish(out/'input.jsonl',out/'receipt.jsonl',out/'release')
        assert manifest['rows']==1
        tokenizer=AutoTokenizer.from_pretrained('/root/autodl-fs/models/Qwen3.5-9B',local_files_only=True)
        config=OmegaConf.create({'messages_key':'messages','max_length':8192,'pad_mode':'no_padding',
                                'truncation':'error','ignore_input_ids_mismatch':True})
        ds=GlasseekSFTDataset(str(out/'release/train.parquet'),tokenizer,config)
        example=ds[0]
        parts=[]; expected=[]
        for i,m in enumerate(ds.messages[0]):
            ids,mask,attention,_=ds._process_single_message(i,m,ds.messages[0],ds.tools[0] if i==0 else None,False)
            assert int(mask.sum())>0 if i==4 else int(mask.sum())==0
            parts.append(ids);expected.append(mask)
        assert torch.equal(example['input_ids'],torch.cat(parts))
        assert torch.equal(example['loss_mask'],torch.cat(expected))
        assert 'intermediate message stays in context' in tokenizer.decode(example['input_ids'])
        print(json.dumps({'tokens':len(example['input_ids']),'loss_tokens':int(example['loss_mask'].sum()),
                          'masked_assistant_retained':True,'real_loader':True}))
    ''')
    env={**os.environ,'PYTHONNOUSERSITE':'1','HF_HUB_OFFLINE':'1','GLASSEEK_URL_MODE':'raw',
         'PYTHONPATH':':'.join(str(root/p) for p in ('Shared','RL_training','.cache/verl-v091','scripts/trajectory_preprocess'))}
    result=subprocess.run([str(python),'-c',script,str(tmp_path)],env=env,cwd=root,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
