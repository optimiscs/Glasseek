"""Archive prior outcomes and refresh official checkpoints from native history."""
import asyncio,json,shutil,sqlite3,sys
from pathlib import Path

ROOT=Path('/root/autodl-tmp/Glasseek')
OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'Shared'),str(ROOT/'Evaluation_backend'),str(ROOT/'Evaluation')]
from glasseek_shared.context.official import activate
activate()
from clm_harness.utils.resume import Checkpointer
from glasseek_shared.context.official.local import LocalEnvironment
from glasseek_shared.context.official.tool_rounds import recorded_tool_rounds
from browse_comp_eval.resume_checkpoint import write_checkpoint
from hermes_state import SessionDB

async def main():
    inventory=json.loads((OUT/'inventory.json').read_text())
    planned=[]
    for record in inventory:
        if record['status']=='success':continue
        work=ROOT/record['workspace'];source=ROOT/record['path']
        cfg=json.loads((work/'official-clm-request.json').read_text())
        official=Path(cfg['output'])
        archive=OUT/'previous'/record['q']
        archive.mkdir(parents=True,exist_ok=True)
        if (archive/'workspace').exists():raise RuntimeError('Already prepared '+record['q'])
        shutil.copytree(work,archive/'workspace',symlinks=True)
        dbfile=ROOT/'Evaluation/data/runtime/hermes/profiles'/cfg['session_id'].removeprefix('clm-')/'state.db'
        with sqlite3.connect(dbfile) as native,sqlite3.connect(archive/'state.db') as backup:
            native.backup(backup)
            rounds=recorded_tool_rounds(native)
        db=SessionDB(dbfile)
        try:
            sid=db.get_compression_tip(cfg['session_id'])
            messages=db.get_messages_as_conversation(sid)
        finally:db.close()
        # Official replay handles any trailing unanswered action. Do not invent results.
        env=LocalEnvironment(official/'workspace')
        checkpointer=Checkpointer(official/'agent',interval_s=30,source=env._workdir)
        if not await checkpointer.save(env,messages):raise RuntimeError('Checkpoint failed '+record['q'])
        write_checkpoint(work,run_id=record['q']+'__r1',profile=cfg['session_id'].removeprefix('clm-'),
                         started_at=cfg['started_at'])
        source.replace(archive/'prediction.json')
        judge=source.parents[2]/'judge_official'/record['q']/(record['q']+'__r1.json')
        if judge.exists():judge.replace(archive/'judge.json')
        planned.append(dict(question_id=record['q'],previous_status=record['status'],
            rounds_used=rounds,rounds_remaining=max(0,120-rounds),messages=len(messages),
            checkpoint=str(official),archive=str(archive)))
        print(record['q'],f'{rounds}/120 tool rounds; checkpoint refreshed',flush=True)
    (OUT/'resume-plan.json').write_text(json.dumps({'preserved_scored':18,'resuming':planned},indent=2))

if __name__=='__main__':asyncio.run(main())
