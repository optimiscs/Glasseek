"""Harbor environment adapter over Glasseek's existing local terminal backend."""
import asyncio
import fcntl
import json
from pathlib import Path
import shutil
import sys
import uuid
from types import SimpleNamespace
from . import ROOT

class LocalEnvironment:
    def __init__(self, workspace):
        self._workdir=str(Path(workspace).resolve())
        self.name='clm-local-'+uuid.uuid4().hex[:12]
        self._lock=None
        self.backend=None

    async def start(self):
        Path(self._workdir).mkdir(parents=True,exist_ok=True)
        lock=(Path(self._workdir)/'.clm-task.lock').open('a')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close();raise RuntimeError('This CLM task workspace is already in use; choose another workspace')
        self._lock=lock
        Path(self._workdir).mkdir(parents=True,exist_ok=True)
        try:
            sys.path.insert(0,str(ROOT/'Evaluation_backend'))
            from tools.environments.local import LocalEnvironment as ExistingLocalEnvironment
            self.backend=ExistingLocalEnvironment(cwd=self._workdir,timeout=120)
        except BaseException:
            lock.close();self._lock=None
            raise

    async def stop(self):
        try:
            if self.backend is not None:
                cleanup=getattr(self.backend,'cleanup',None)
                if cleanup:await asyncio.to_thread(cleanup)
        finally:
            if self._lock is not None:
                self._lock.close();self._lock=None

    async def exec(self, command, timeout_sec=120):
        result=await asyncio.to_thread(self.backend.execute,command,cwd=self._workdir,timeout=timeout_sec)
        return SimpleNamespace(stdout=result.get('output',''),stderr='',return_code=result.get('returncode',0))

    async def upload_file(self, source_path, target_path):
        p=Path(target_path);p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source_path,p)

    async def download_file(self, source_path, target_path):
        await self.upload_file(source_path,target_path)

    async def download_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir,target_dir,dirs_exist_ok=True,symlinks=True)

    async def upload_dir(self, source_dir, target_dir):
        shutil.copytree(source_dir,target_dir,dirs_exist_ok=True,symlinks=True)
        metadata=Path(target_dir)/'.clm/paths.json'
        if metadata.exists():
            paths=json.loads(metadata.read_text())
            old=paths.get('workspace')
            if old and old!=self._workdir:
                for name in ('cwd','env'):
                    state=Path(target_dir)/'.clm/bash'/name
                    if state.exists():state.write_text(state.read_text().replace(old,self._workdir))
                metadata.write_text(json.dumps({k:v.replace(old,self._workdir) for k,v in paths.items()},indent=2))
