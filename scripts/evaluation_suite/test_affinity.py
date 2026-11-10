import json
from affinity import AffinityScheduler


def plans(names):
    return [(name,['python','native_eval.py','--model','same','--base-urls','http://old/v1',
                   '--workers-per-endpoint','16','--num-workers','16'], {}) for name in names]


def test_whole_dataset_affinity_then_hot_add_only_pending(tmp_path):
    endpoints=['http://gpu0/v1','http://gpu1/v1']
    scheduler=AffinityScheduler(endpoints,plans(['a','b']),{'a':100,'b':10},tmp_path,16)
    ready=scheduler.ready()
    assert len(ready)==2
    assert all('--base-urls' not in cmd for _,cmd,_ in ready)
    assert all(len(e)==1 for e in scheduler.active.values())
    scheduler.finished('b')
    # Do not join a queue containing only in-flight tasks.
    p=tmp_path/'scheduling/a.json';p.parent.mkdir();p.write_text('{"unscheduled":0}')
    assert scheduler.ready()==[] and scheduler.owner[endpoints[1]] is None
    p.write_text('{"unscheduled":50}')
    assert scheduler.ready()==[]
    assert scheduler.active['a']==endpoints
    registry=json.loads((tmp_path/'endpoint-registries/a.json').read_text())
    assert [e['base_url'] for e in registry['models']['same']]==endpoints
    assert sum(16 for owner in scheduler.owner.values() if owner)==32
    # Repeated ticks never hot-register the same GPU twice.
    scheduler.ready()
    assert len(scheduler.active['a'])==2
    scheduler.finished('a')
    assert all(owner is None for owner in scheduler.owner.values()) and not scheduler.pending


def test_idle_gpu_steals_whole_queued_dataset_before_splitting(tmp_path):
    scheduler=AffinityScheduler(['http://0','http://1'],plans(['a','b','c']),{'a':100,'b':50,'c':40},tmp_path,16)
    assert len(scheduler.ready())==2
    scheduler.finished('a')
    ready=scheduler.ready()
    assert [r[0] for r in ready]==['c']
    assert len(scheduler.active['c'])==1


def test_orchestrator_uses_per_card_limit_and_adds_idle_card(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import orchestrate
    root=tmp_path/'run';root.mkdir()
    args=SimpleNamespace(output_root=tmp_path,run_id='run',workers=16,phase='infer',
                         score_schedule='off',affinity_endpoints='http://0/v1,http://1/v1')
    joined=[]
    class Process:
        pid=123
        def __init__(self,cmd,env,**unused):
            self.path=__import__('pathlib').Path(cmd[cmd.index('--endpoint-registry')+1])
            self.name=self.path.stem;self.polls=0
            assert cmd[cmd.index('--workers-per-endpoint')+1]=='16'
            p=__import__('pathlib').Path(env['GLASSEEK_SCHEDULING_STATE']);p.parent.mkdir(exist_ok=True)
            p.write_text('{"unscheduled":50}')
        def poll(self):
            self.polls+=1
            records=json.loads(self.path.read_text())['models']['same']
            if self.name=='a' and len(records)==2:joined.append(records)
            return 0 if self.polls >= (5 if self.name=='a' else 1) else None
        def terminate(self):pass
    monkeypatch.setattr(orchestrate.subprocess,'Popen',Process)
    monkeypatch.setattr(orchestrate,'write_report',lambda *a: None)
    monkeypatch.setattr(orchestrate.time,'sleep',lambda *a: None)
    orchestrate.execute(args,plans(['a','b']),[],['a','b'],{'a':100,'b':10})
    assert joined
