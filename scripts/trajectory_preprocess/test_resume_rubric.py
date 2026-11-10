import json
from collections import Counter
from types import SimpleNamespace

from resume_rubric import balanced_selection, extend_pool
from rubric_batch import connect, pack


def test_balances_cumulative_counts_and_redistributes_exhausted_pool():
    candidates=[(name,i,f'{name}{i}') for name,n in [('A',20),('B',20),('C',2)] for i in range(n)]
    picked=balanced_selection(candidates,{'A':10,'B':2},16)
    assert Counter(x[0] for x in picked)=={'A':3,'B':11,'C':2}
    assert picked==sorted(picked)
    assert balanced_selection(list(reversed(candidates)),{'A':10,'B':2},16)==picked
    assert len(balanced_selection(candidates,{},100))==42


def test_extension_reselects_queue_skips_results_and_appends(tmp_path,monkeypatch):
    import resume_rubric as module
    pool=tmp_path/'pool';out=tmp_path/'out'
    pool.mkdir();out.mkdir()
    src=connect(pool/'batch.sqlite');dst=connect(out/'batch.sqlite')
    rows=[]
    for i,name in enumerate(['A','A','A','A','B','B','B']):
        row=(i,str(i),pack({'source':name}),pack({}),'artifact',i)
        rows.append(row);src.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',row)
    for row in rows[:4]:dst.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',row)
    dst.execute('INSERT INTO results VALUES (?,?,?)',('0','completed',json.dumps({'id':'0','source':'A'})))
    dst.execute('INSERT INTO results VALUES (?,?,?)',('1','inflight',None))
    src.commit();dst.commit()
    (out/'config.json').write_text(json.dumps({'scoring_mode':'numeric','generation':{'seed':42,'temperature':0}}))
    (out/'results.jsonl').write_text('{"id":"0"}\n')
    (out/'reused-128.jsonl').write_text('{"id":"anchor","source":"A"}\n')
    monkeypatch.setattr(module,'fingerprint',lambda:'test')
    monkeypatch.setattr(module,'public_config',lambda x:x)
    monkeypatch.setattr(module,'run',lambda *args:None)
    args=SimpleNamespace(output=out,full_pool=pool,limit=4,dry_run=True)
    extend_pool(args)
    assert dst.execute('SELECT count(*) FROM inputs').fetchone()[0]==4
    assert (out/'results.jsonl').read_text()=='{"id":"0"}\n'
    args.dry_run=False
    extend_pool(args)
    assert set(x[0] for x in dst.execute('SELECT id FROM inputs'))=={'0','1','4','5'}
    assert dst.execute("SELECT result FROM results WHERE id='0'").fetchone()[0]==json.dumps({'id':'0','source':'A'})
    assert json.loads((out/'results.jsonl').read_text().splitlines()[1])['decision']=='interrupted_discarded'
    extend_pool(args)
    assert set(x[0] for x in dst.execute('SELECT id FROM inputs'))=={'0','1','4','5'}
    assert len((out/'results.jsonl').read_text().splitlines())==2
    plan=json.loads((out/'allocation.json').read_text())
    assert plan['existing_total']==2 and plan['scheduled']==2 and plan['projected_total']==4
    args.limit=1
    extend_pool(args)
    assert set(x[0] for x in dst.execute('SELECT id FROM inputs'))=={'0','1'}
    assert json.loads((out/'allocation.json').read_text())['scheduled']==0


def test_index_reuse_and_wal_invalidation(tmp_path,monkeypatch,capsys):
    import resume_rubric as module
    path=tmp_path/'batch.sqlite'
    db=connect(path)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(0,'a',pack({'source':'A'}),pack({}),'x',1))
    db.commit()
    assert module.pool_index(path)==[('A',0,'a')]
    original=module.unpack
    monkeypatch.setattr(module,'unpack',lambda _: (_ for _ in ()).throw(AssertionError('must use index')))
    assert module.pool_index(path)==[('A',0,'a')]
    monkeypatch.setattr(module,'unpack',original)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(1,'b',pack({'source':'B'}),pack({}),'x',2))
    db.commit()
    assert module.pool_index(path)==[('A',0,'a'),('B',1,'b')]
    assert 'Index hit' in capsys.readouterr().err


def test_failed_index_build_keeps_previous_cache(tmp_path,monkeypatch):
    import resume_rubric as module
    import pytest
    path=tmp_path/'batch.sqlite';db=connect(path)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(0,'a',pack({'source':'A'}),pack({}),'x',1))
    db.commit();module.pool_index(path)
    previous=(tmp_path/'sample-index.sqlite').read_bytes()
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(1,'b',pack({'source':'B'}),pack({}),'x',2))
    db.commit()
    monkeypatch.setattr(module,'unpack',lambda _: (_ for _ in ()).throw(ValueError('broken')))
    with pytest.raises(ValueError,match='broken'):module.pool_index(path)
    assert (tmp_path/'sample-index.sqlite').read_bytes()==previous
    assert not list(tmp_path.glob('sample-index-*.tmp'))


def test_index_rejects_source_write_during_scan(tmp_path,monkeypatch):
    import resume_rubric as module
    import pytest
    path=tmp_path/'batch.sqlite';db=connect(path)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(0,'a',pack({'source':'A'}),pack({}),'x',1))
    db.commit()
    original=module.unpack
    def concurrent_write(blob):
        db.execute('UPDATE inputs SET sample=? WHERE id=?',(pack({'source':'changed'}),'a'))
        db.commit()
        return original(blob)
    monkeypatch.setattr(module,'unpack',concurrent_write)
    with pytest.raises(RuntimeError,match='changed during indexing'):module.pool_index(path)
    assert not (tmp_path/'sample-index.sqlite').exists()


def test_index_survives_checkpoint_touch_and_unrelated_writes(tmp_path,monkeypatch):
    import os
    import resume_rubric as module
    path=tmp_path/'batch.sqlite';db=connect(path)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(0,'a',pack({'source':'A'}),pack({}),'x',1))
    db.commit();assert module.pool_index(path)==[('A',0,'a')]
    monkeypatch.setattr(module,'unpack',lambda _: (_ for _ in ()).throw(AssertionError('unexpected rebuild')))
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    db.execute("INSERT INTO meta VALUES ('unrelated','changed')");db.commit()
    os.utime(path,None)
    assert module.pool_index(path)==[('A',0,'a')]
    db.close()
    assert module.pool_index(path)==[('A',0,'a')]


def test_index_rebuilds_same_count_source_change(tmp_path):
    import resume_rubric as module
    path=tmp_path/'batch.sqlite';db=connect(path)
    db.execute('INSERT INTO inputs VALUES (?,?,?,?,?,?)',(0,'a',pack({'source':'A'}),pack({}),'x',1))
    db.commit();module.pool_index(path)
    db.execute('UPDATE inputs SET sample=? WHERE id=?',(pack({'source':'B'}),'a'));db.commit()
    assert module.pool_index(path)==[('B',0,'a')]
