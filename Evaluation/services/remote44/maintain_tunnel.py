"""Keep a loopback-only SSH forward alive; optionally adopt our existing SSH."""
import os
import pathlib
import subprocess
import sys
import time

local, remote = int(sys.argv[1]), int(sys.argv[2])
adopted = int(sys.argv[3]) if len(sys.argv)>3 else 0
root=pathlib.Path('/root/autodl-tmp/Glasseek/Evaluation/data/runtime')

def is_adopted_alive(pid):
 try:
  proc=pathlib.Path(f'/proc/{pid}')
  cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ')
  state=proc.joinpath('stat').read_text().split(') ')[1][0]
  return state!='Z' and b'ssh ' in cmd and f'127.0.0.1:{local}:127.0.0.1:{remote}'.encode() in cmd
 except (OSError,IndexError):return False

while adopted and is_adopted_alive(adopted):time.sleep(2)
while True:
 cmd=['ssh','-o','BatchMode=yes','-o','ExitOnForwardFailure=yes','-o','ConnectTimeout=10',
      '-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','-N','-L',f'127.0.0.1:{local}:127.0.0.1:{remote}','44']
 print(time.strftime('%FT%T%z'),'connecting',local,remote,flush=True)
 try:
  p=subprocess.Popen(cmd);(root/f'ssh44-tunnel-{local}.pid').write_text(str(p.pid));rc=p.wait()
  print('ssh exited',rc,flush=True)
 except OSError as e:print('ssh failed',type(e).__name__,flush=True)
 time.sleep(3)
