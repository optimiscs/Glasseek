"""Share byte-identical inactive profile binaries and provision a common PATH binary."""
import hashlib,os,stat
from pathlib import Path
root=Path('/root/autodl-tmp/Glasseek/Evaluation/data/runtime/hermes')
expected='f85be2e601d26e2995bc85c9df7a85a2fff141f7166153b7e0f82a037a2a8835'
profiles=root/'profiles'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def active():
 names=set()
 for p in Path('/proc').glob('[0-9]*'):
  try:
   args=p.joinpath('cmdline').read_bytes().split(b'\0')
   if b'--profile' in args:names.add(args[args.index(b'--profile')+1].decode(errors='replace'))
  except (OSError,ValueError,IndexError):pass
 return names
candidates=[p for p in profiles.glob('*/bin/tirith') if p.is_file() and not p.is_symlink() and p.stat().st_size==46144280]
source=next(p for p in candidates if digest(p)==expected and stat.S_IMODE(p.stat().st_mode)==0o755)
shared=root/'bin/tirith';shared.parent.mkdir(parents=True,exist_ok=True)
if shared.exists():
 if digest(shared)!=expected: raise SystemExit('Existing shared tirith differs; no changes made')
else: os.link(source,shared)
pathlink=Path('/usr/local/bin/tirith')
if pathlink.exists() or pathlink.is_symlink():
 if not pathlink.exists() or digest(pathlink)!=expected: raise SystemExit('Existing PATH tirith differs; no profiles changed')
else: pathlink.symlink_to(shared)
seen=active();replaced=skipped=0
for target in candidates:
 try:
  if target.parent.parent.name in seen or target.parent.parent.name in active():skipped+=1;continue
  before=target.stat()
  if before.st_ino==shared.stat().st_ino:continue
  if stat.S_IMODE(before.st_mode)!=0o755 or (before.st_uid,before.st_gid)!=(0,0) or digest(target)!=expected:continue
  current=target.stat()
  if (before.st_ino,before.st_size,before.st_mtime_ns)!=(current.st_ino,current.st_size,current.st_mtime_ns):continue
  temp=target.with_name('.tirith-dedupe-'+str(os.getpid()))
  os.link(shared,temp)
  if target.parent.parent.name in active():temp.unlink();skipped+=1;continue
  os.replace(temp,target);replaced+=1
 except OSError as exc: print('skip',target,type(exc).__name__,flush=True)
print('shared',shared,'sha256',expected,'deduplicated',replaced,'skipped_active',skipped,'nominal_GiB_saved',round(replaced*46144280/2**30,2),flush=True)
