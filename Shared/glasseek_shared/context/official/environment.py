"""Harbor-compatible CPU Docker environment on the configured SSH host."""
import asyncio
import base64
import io
from pathlib import Path
import shlex
import tarfile
from types import SimpleNamespace
import uuid

class DockerEnvironment:
    _workdir='/workspace'
    def __init__(self, host='44', image='709_model:final', name=None):
        self.host=host;self.image=image;self.name=name or 'glasseek-clm-'+uuid.uuid4().hex[:12]

    async def _run(self, argv, data=None, timeout=180):
        p=await asyncio.create_subprocess_exec('ssh','-o','BatchMode=yes',self.host,
            shlex.join(argv),stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        try:out,err=await asyncio.wait_for(p.communicate(data),timeout)
        except BaseException:
            p.kill();await p.wait();raise
        if p.returncode:raise RuntimeError(err.decode(errors='replace')[-2000:])
        return out

    async def start(self):
        await self._run(['docker','run','-d','--name',self.name,'--network','host','--cpus','2','--memory','2g',
            '--cap-drop','ALL','--security-opt','no-new-privileges','--entrypoint','bash',self.image,'-c','mkdir -p /workspace; sleep infinity'])

    async def stop(self):
        await self._run(['docker','rm','-f',self.name])

    async def exec(self, command, timeout_sec=120):
        script='import subprocess,json; p=subprocess.run(["bash","-lc",'+repr(command)+'],capture_output=True,text=True,timeout='+str(timeout_sec)+'); print(json.dumps(dict(stdout=p.stdout,stderr=p.stderr,return_code=p.returncode)))'
        out=await self._run(['docker','exec','-w','/workspace',self.name,'python3','-c',script],timeout=timeout_sec+15)
        import json
        return SimpleNamespace(**json.loads(out))

    async def upload_file(self, source_path, target_path):
        script='import sys,pathlib;p=pathlib.Path('+repr(target_path)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(sys.stdin.buffer.read())'
        await self._run(['docker','exec','-i',self.name,'python3','-c',script],Path(source_path).read_bytes())

    async def download_file(self, source_path, target_path):
        out=await self._run(['docker','exec',self.name,'python3','-c','import pathlib,sys;sys.stdout.buffer.write(pathlib.Path('+repr(source_path)+').read_bytes())'])
        p=Path(target_path);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(out)

    async def download_dir(self, source_dir, target_dir):
        raw=await self._run(['docker','exec',self.name,'tar','-C',source_dir,'-cf','-','.'])
        Path(target_dir).mkdir(parents=True,exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:archive.extractall(target_dir,filter='data')

    async def upload_dir(self, source_dir, target_dir):
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for p in Path(source_dir).iterdir():archive.add(p,arcname=p.name)
        await self.exec('mkdir -p '+shlex.quote(target_dir))
        await self._run(['docker','exec','-i',self.name,'tar','-C',target_dir,'-xf','-'],stream.getvalue())
