import json
from affinity import AffinityScheduler, atomic_json
from glasseek_shared.endpoint_slots import try_acquire, snapshot


def plans(names):
    return [(n,['python','native_eval.py','--model','same','--base-urls','http://old/v1',
                '--workers-per-endpoint','16','--num-workers','16'],{}) for n in names]


def progress(scheduler, name, pending, in_flight):
    atomic_json(scheduler.root/'scheduling'/f'{name}.json',{'unscheduled':pending,'in_flight':in_flight})


def test_small_dataset_does_not_reserve_whole_gpu(tmp_path):
    s=AffinityScheduler(['gpu0','gpu1'],plans(['large','small']),{'large':100,'small':2},tmp_path,16)
    try:
        assert len(s.ready())==2
        assert s.quotas['gpu0']=={'large':16}
        assert s.quotas['gpu1']=={'small':2,'large':14}
        for name in ['large','small']:
            entries=json.loads(s.registry(name).read_text())['models']['same']
            assert len(entries)==2 and all('worker_limit' in e for e in entries)
        assert sum(sum(q.values()) for q in s.quotas.values())==32
        original=json.loads(json.dumps(s.events[0]))
        s.finished('small')
        assert s.events[0]==original
    finally:s.close()


def test_backfill_while_old_tasks_stay_in_flight(tmp_path):
    s=AffinityScheduler(['gpu0','gpu1'],plans(['a','b','c']),{'a':100,'b':80,'c':50},tmp_path,16)
    leases=[]
    try:
        s.ready()
        leases=[try_acquire(s.slots,'gpu0','a',16) for _ in range(2)]
        assert all(leases)
        progress(s,'a',0,2)
        progress(s,'b',80,0)
        ready=s.ready()
        assert s.quotas['gpu0']['a']==2
        assert s.quotas['gpu0']['c']==14
        assert [r[0] for r in ready]==['c']
        assert snapshot(s.slots,['gpu0','gpu1'],16)['gpu0']=={'a':2}
        assert try_acquire(s.slots,'gpu0','a',16) is None
        lease=try_acquire(s.slots,'gpu0','c',16);assert lease;leases.append(lease)
    finally:
        for lease in leases:lease.close()
        s.close()


def test_current_dataset_replenished_before_other_waiters(tmp_path):
    s=AffinityScheduler(['gpu0'],plans(['a','b']),{'a':50,'b':20},tmp_path,16)
    try:
        assert [r[0] for r in s.ready()]==['a']
        progress(s,'a',30,0)
        assert s.ready()==[] and s.quotas['gpu0']=={'a':16}
        progress(s,'a',0,0)
        assert [r[0] for r in s.ready()]==['b']
        assert s.quotas['gpu0']=={'b':16}
    finally:s.close()


def test_duplicate_coordinator_rejected(tmp_path):
    import pytest
    s=AffinityScheduler(['gpu0'],plans(['a']),{'a':10},tmp_path,16)
    try:
        with pytest.raises(ValueError,match='already owns'):
            AffinityScheduler(['gpu0'],plans(['a']),{'a':10},tmp_path,16)
    finally:s.close()


def test_orchestrator_launches_backfill_without_waiting_for_dataset_exit(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import orchestrate
    root=tmp_path/'run';root.mkdir()
    args=SimpleNamespace(output_root=tmp_path,run_id='run',workers=16,phase='infer',
                         score_schedule='off',affinity_endpoints='gpu0,gpu1')
    launched=[]
    class Process:
        pid=123
        def __init__(self,cmd,env,**unused):
            from pathlib import Path
            self.name=Path(cmd[cmd.index('--endpoint-registry')+1]).stem
            self.polls=0;launched.append(self.name)
            atomic_json(env['GLASSEEK_SCHEDULING_STATE'],{'unscheduled':0,'in_flight':2})
        def poll(self):
            self.polls+=1
            if self.name=='a' and self.polls==2:assert 'c' in launched
            return 0 if self.polls>=3 else None
        def terminate(self):pass
    monkeypatch.setattr(orchestrate.subprocess,'Popen',Process)
    monkeypatch.setattr(orchestrate,'write_report',lambda *a:None)
    monkeypatch.setattr(orchestrate.time,'sleep',lambda *a:None)
    orchestrate.execute(args,plans(['a','b','c']),[],['a','b','c'],{'a':100,'b':80,'c':50})
    assert set(launched)=={'a','b','c'}


def test_unhealthy_endpoint_does_not_strand_pending_dataset(tmp_path):
    s=AffinityScheduler(['gpu0','gpu1'],plans(['a','b']),{'a':100,'b':2},tmp_path,16)
    try:
        s.ready()
        for name in s.active:
            atomic_json(s.root/'scheduling'/f'{name}.json',{'unscheduled':100 if name=='a' else 0,
                'in_flight':0, 'endpoint_health':[{'base_url':'gpu0','state':'unhealthy'},
                                                {'base_url':'gpu1','state':'healthy'}]})
        s.ready()
        assert s.quotas['gpu0']=={} and s.quotas['gpu1']=={'a':16}
    finally:s.close()
