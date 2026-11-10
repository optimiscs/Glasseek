import asyncio,json,tempfile
from pathlib import Path
from glasseek_shared.context.official import activate,COMMIT
from glasseek_shared.context.official.local import LocalEnvironment
activate()
from clm_harness.context_env.env import ContextEnv
from clm_harness.context_env.paths import Paths
from clm_harness.utils.budget import BudgetController
from clm_harness.utils.resume import Checkpointer,ResumeState

async def main():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);logs=root/'trial/agent';logs.mkdir(parents=True)
  messages=[{'role':'system','content':'pinned system'},{'role':'user','content':'pinned task'},
      {'role':'assistant','content':'Repeated research '+('old evidence '*300)}]
  budget=BudgetController(10000,reserve_tokens=100)
  context=ContextEnv(paths=Paths(None),budget=budget,persistent_bash=True,allow_edit_growth=False,context_budget_tokens=10000)
  e=LocalEnvironment(root/'first')
  try:
   await e.start();await context.setup_dirs(e)
   command="python3 - <<'END'\nfrom pathlib import Path\np=Path('/tmp/.live_ctx/LIVE_CTX_MAIN.txt');s=p.read_text();p.write_text(s.split('\\n',1)[0]+'\\nRetained evidence U1; unresolved question remains.\\n')\nEND"
   result=await context.step(command,messages,environment=e)
   assert result.ctx_changed and messages[:2]==[{'role':'system','content':'pinned system'},{'role':'user','content':'pinned task'}]
   await e.exec('printf READY > answer.txt')
   checkpointer=Checkpointer(logs,interval_s=1)
   assert await checkpointer.save(e,messages)
  finally:await e.stop()
  second=LocalEnvironment(root/'second')
  try:
   await second.start();resume=ResumeState(root/'trial');await resume.restore(second)
   restored=resume.replay(messages[:2],budget_tokens=10000)
   assert (root/'second/answer.txt').read_text()=='READY'
   assert any('Retained evidence U1' in str(m.get('content')) for m in restored)
  finally:await second.stop()
  print(json.dumps({'official_commit':COMMIT,'backend':'existing Glasseek local terminal','mirror_edit_applied':True,'pinned_prefix_preserved':True,'official_checkpoint_saved':True,'new_environment_workspace_restored':True,'context_restored':True,'docker_required':False}))
asyncio.run(main())
