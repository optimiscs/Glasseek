import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/validation'))
from search_pool_load import Sandbox


def test_four_key_probe_isolated_restored_and_usage_merged(tmp_path):
    with Sandbox(tmp_path/'mock',keys=[f'fake-{i}' for i in range(6)],replicas=1) as s:
        before=s.redis.hgetall(s.prefix+'config')
        output=tmp_path/'result.json'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/validation/search_pool_key_rps.py'),
            '--seconds','1','--output',str(output)],env=s.env,capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr
        report=json.loads(output.read_text())
        assert len(report['keys'])==4
        assert all(k['success']==8 and k['upstream_attempts']==8 for k in report['keys'])
        assert s.health()['eligible_keys']==6
        assert s.redis.hgetall(s.prefix+'config')==before
        assert len(s.stats()['rows'])==32
        assert sum(v.get('confirmed_credits',0) for v in s.health()['lifetime_usage'].values())==32
        assert report['production_keys_restored']
