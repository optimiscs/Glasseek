import json
import multiprocessing as mp
import os
import time

from glasseek_shared.endpoint_slots import try_acquire, snapshot


def worker(root, dataset, ready, release):
    leases=[]
    for _ in range(24):
        lease=try_acquire(root,'gpu',dataset,16)
        if lease:leases.append(lease)
    ready.put((dataset,len(leases)))
    release.wait(10)
    for lease in leases:lease.close()


def test_processes_share_one_sixteen_slot_limit(tmp_path):
    # Even deliberately overlapping stale quotas cannot exceed the physical cap.
    (tmp_path/'allocation.json').write_text(json.dumps({'capacity':16,'quotas':{'gpu':{'a':16,'b':16,'c':16}}}))
    ctx=mp.get_context('spawn');ready=ctx.Queue();release=ctx.Event()
    processes=[ctx.Process(target=worker,args=(str(tmp_path),n,ready,release)) for n in ['a','b','c']]
    try:
        for p in processes:p.start()
        acquired=[ready.get(timeout=15) for _ in processes]
        assert sum(n for _,n in acquired)==16
        assert sum(snapshot(tmp_path,['gpu'],16)['gpu'].values())==16
        assert try_acquire(tmp_path,'gpu','a',16) is None
    finally:
        release.set()
        for p in processes:
            p.join(15)
            if p.is_alive():p.kill();p.join()
    assert snapshot(tmp_path,['gpu'],16)=={'gpu':{}}


def test_process_death_releases_slot_without_expiring_live_tasks(tmp_path):
    (tmp_path/'allocation.json').write_text(json.dumps({'capacity':16,'quotas':{'gpu':{'a':16,'b':16}}}))
    ctx=mp.get_context('spawn');ready=ctx.Queue();release=ctx.Event()
    p=ctx.Process(target=worker,args=(str(tmp_path),'a',ready,release));p.start()
    try:
        assert ready.get(timeout=15)[1]==16
        p.kill();p.join(10)
        lease=try_acquire(tmp_path,'gpu','b',16)
        assert lease is not None
        lease.close()
    finally:
        if p.is_alive():p.kill();p.join()


def test_actual_endpoint_pools_share_two_gpu_limits(tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from browse_comp_eval.hermes_client import MultiEndpointHermesClient, HermesConfig
    endpoints=['http://gpu0/v1','http://gpu1/v1']
    (tmp_path/'allocation.json').write_text(json.dumps({'capacity':16,
        'quotas':{e:{'a':16,'b':16} for e in endpoints}}))
    monkeypatch.setenv('GLASSEEK_ENDPOINT_SLOTS',str(tmp_path))
    monkeypatch.setenv('GLASSEEK_ENDPOINT_CAPACITY','16')
    pools=[]
    for dataset in ['a','b']:
        monkeypatch.setenv('GLASSEEK_SLOT_DATASET',dataset)
        config=HermesConfig('unused','local','', 'model','custom','test',dataset,1,1,[],[])
        pools.append(MultiEndpointHermesClient(config,endpoints,16,
            client_factory=lambda config:SimpleNamespace(config=config)))
    lock=threading.Lock();active=[0,0];peak=[0,0];peak_total=0
    def task(i):
        nonlocal peak_total
        with pools[i%2].bind() as bound:
            route=bound.route_index
            with lock:
                active[route]+=1;peak[route]=max(peak[route],active[route]);peak_total=max(peak_total,sum(active))
                assert max(active)<=16 and sum(active)<=32
            time.sleep(.15)
            with lock:active[route]-=1
    with ThreadPoolExecutor(max_workers=64) as executor:
        list(executor.map(task,range(96)))
    assert peak==[16,16] and peak_total==32
    assert snapshot(tmp_path,endpoints,16)=={e:{} for e in endpoints}


def test_zero_quota_refresh_does_not_retire_or_cancel_active_route(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from browse_comp_eval.hermes_client import MultiEndpointHermesClient, HermesConfig
    from browse_comp_eval.endpoint_health import EndpointWatchdogConfig
    registry=tmp_path/'registry.json'
    def write(limit):
        registry.write_text(json.dumps({'version':1,'models':{'model':[
            {'base_url':'http://gpu/v1','worker_limit':limit}]}}))
    write(16)
    config=HermesConfig('unused','local','', 'model','custom','test','a',1,1,[],[])
    cancel=Mock()
    pool=MultiEndpointHermesClient(config,[],16,endpoint_registry=registry,
        watchdog_config=EndpointWatchdogConfig(enabled=True),
        client_factory=lambda config:SimpleNamespace(config=config,cancel_active=cancel))
    pool.health_manager.config=EndpointWatchdogConfig(enabled=False)
    pool._refresh_endpoint_registry()
    with pool.bind() as bound:
        generation=bound.generation
        write(0);pool._refresh_endpoint_registry()
        assert pool._active==[1] and pool.worker_limits==[0]
        assert pool.health_manager.generation(0)==generation
        assert not pool.endpoint_health_snapshot()[0]['retired']
        cancel.assert_not_called()
    assert pool._active==[0] and pool.scheduling_capacity()==0
