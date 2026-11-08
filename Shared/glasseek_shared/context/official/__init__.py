"""Glasseek environment adapters for unmodified upstream CLM."""
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[4]
UPSTREAM=ROOT/'third_party/context-language-models'
COMMIT='483198229ab7b90c0eed4651306ab3b3505aaf50'
def activate():
    if subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()!=COMMIT:
        raise RuntimeError('Official CLM commit mismatch')
    if subprocess.check_output(['git','-C',str(UPSTREAM),'diff','--name-only','HEAD'],text=True).strip():
        raise RuntimeError('Official CLM source modified')
    sys.path.insert(0,str(UPSTREAM/'clm'))
