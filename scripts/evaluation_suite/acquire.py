#!/usr/bin/env python3
"""Fetch pinned official data and create clean, isolated author evaluator checkouts."""
import argparse
import json
import subprocess
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from suite import ROOT, dump, sha

QUEST=ROOT.parent/'Quest'
OFFICIAL=QUEST/'workspace/strict-reproduction/official-dataset-eval-20261028/sources'
PAPER=QUEST/'workspace/strict-reproduction/paper8-qwen35-8-20261028-r1/sources'
REPOS={
 'simple-evals':(OFFICIAL/'simple-evals','652c89d0ca9df547706735883097e9537d40dc47'),
 'hle':(OFFICIAL/'hle','22ed3074b1e7b134bcbc09028d0ba320839b0655'),
 'deep-research-bench':(OFFICIAL/'deep-research-bench','f2735b5c3636759c22f9e936c0232de8bf545b67'),
 'gaia-leaderboard':(QUEST/'workspace/strict-reproduction/official-sources/gaia-leaderboard','9f133d71362e77b3539f1514f31b9c101a545fec'),
 'browsecomp-plus':(PAPER/'browsecomp-plus','046949032b0328319cc9a02663a759ec601d9402'),
 'mind2web2':(PAPER/'mind2web2','1fcbfda98bf5c669daa2ed0a27f1d2f2499475ee'),
 'liveresearchbench':(PAPER/'liveresearchbench','ffb27c9d7a091a0b0f10cbcebc061fc3ba025b63'),
 'widesearch':(PAPER/'widesearch','9825ba7b140b71d81b364793f86dabe4cfed6749'),
}
DATA=[
 ('GAIA-benchmark/GAIA','682dd723ee1e1697e00360edccf2366dc8418dd9',['2023/validation/metadata.parquet','README.md']),
 ('cais/hle','5a81a4c7271a2a2a312b9a690f0c2fde837e4c29',['data/*.parquet','README.md']),
 ('Salesforce/LiveResearchBenchFull','b98de2fcb81ab0726f2d2d441922a201c28d3751',['question_with_checklist/*.parquet','README.md']),
 ('osunlp/Mind2Web-2','54f5b2546c30b86812d9564f4800c0c6f77deb45',['*.csv','README.md','evaluation_scripts/2025_10_23/*.py']),
 ('ByteDance-Seed/WideSearch','6531a7e5b497d44c8912407e0cb3dc95bd98cc09',['widesearch.jsonl','README.md','widesearch_gold/*.csv']),
]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,default=ROOT/'data/evaluation/suite5pct-v1');a=p.parse_args()
    for name,(src,pin) in REPOS.items():
        dest=a.base/'evaluators'/name
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            subprocess.run(['git','clone','--quiet','--no-hardlinks','--no-checkout',str(src),str(dest)],check=True)
            subprocess.run(['git','-C',str(dest),'checkout','--quiet','--detach',pin],check=True)
        head=subprocess.check_output(['git','-C',str(dest),'rev-parse','HEAD'],text=True).strip()
        dirty=subprocess.check_output(['git','-C',str(dest),'status','--porcelain','--untracked-files=no'],text=True).strip()
        if head!=pin or dirty: raise ValueError('evaluator drift: '+name)
        print('PINNED',name,pin,flush=True)
    for repo,rev,patterns in DATA:
        snapshot_download(repo,repo_type='dataset',revision=rev,allow_patterns=patterns,
                          local_dir=a.base/'sources'/repo.split('/')[-1],max_workers=4)
        print('FETCHED',repo,rev,flush=True)
    local_sources=[(ROOT/'Evaluation/data/runtime/browsecomp-encrypted.csv','BrowseComp/browsecomp-encrypted.csv'),
        (QUEST/'workspace/strict-reproduction/native-multibench-smoke-20261027-r1/assets/browsecomp_plus_decrypted.jsonl','browsecomp-plus/decrypted.jsonl')]
    for src,relative in local_sources:
        dest=a.base/'sources'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(src,dest)
        if sha(src)!=sha(dest):raise ValueError('frozen local source mismatch')
    files={}
    for folder in ['sources','evaluators']:
        for f in sorted((a.base/folder).rglob('*')):
            if f.is_file() and not {'.git','.cache','__pycache__'}.intersection(f.parts):files[str(f.relative_to(a.base))]=sha(f)
    dump(a.base/'assets.lock.json',{'author_pins':{k:v[1] for k,v in REPOS.items()},'datasets':[{ 'repo':r,'revision':s,'patterns':p} for r,s,p in DATA],'files':files})


if __name__=='__main__':main()
