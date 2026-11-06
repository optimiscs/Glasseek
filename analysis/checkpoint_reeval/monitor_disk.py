"""Monitor reevaluation disk usage and safely deduplicate inactive profile binaries."""
import json,os,shutil,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent
previous_action=0.0
while True:
 usage=shutil.disk_usage('/root/autodl-tmp')
 free=usage.free/2**30
 state={'updated_at':time.time(),'free_gib':round(free,2),'total_gib':round(usage.total/2**30,2),'level':'critical' if free<5 else 'low' if free<10 else 'ok'}
 tmp=root/'disk-monitor-state.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,root/'disk-monitor-state.json')
 print(json.dumps(state),flush=True)
 if free<10 and time.monotonic()-previous_action>600:
  subprocess.run([sys.executable,str(root/'deduplicate_tirith.py')],check=False)
  previous_action=time.monotonic()
 time.sleep(60)
