"""Bounded HTTP adapter for Quest's unmodified webpage evidence selector."""
import asyncio
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, os.environ.get('QUEST_INFERENCE_DIR', str(Path(os.environ.get('QUEST_ROOT', Path(__file__).resolve().parents[3] / 'Quest')) / 'inference')))
from retrieval_experiment import select

log = logging.getLogger('quest_retrieval')
app = FastAPI()
capacity = asyncio.Semaphore(int(os.getenv('GLASSEEK_RETRIEVAL_CONCURRENCY', '2')))

class Selection(BaseModel):
    text: str
    goal: str

@app.get('/health')
async def health():
    return {'status': 'ok', 'method': 'bm25_embedding'}

@app.post('/select')
async def select_page(request: Selection):
    if not request.goal.strip() or not request.text.strip():
        raise HTTPException(400, 'text and goal are required')
    await capacity.acquire()
    try:
        future = asyncio.get_running_loop().run_in_executor(
            None, lambda: select(request.text, request.goal, 'bm25_embedding',
                                 embedding_url=os.environ['VISIT_EMBEDDING_URL']))
    except BaseException:
        capacity.release()
        raise
    future.add_done_callback(lambda _: capacity.release())
    try:
        selected, audit = await asyncio.shield(future)
    except Exception as exc:
        log.warning('Quest retrieval failed: %s', type(exc).__name__)
        raise HTTPException(503, 'Quest retrieval failed') from exc
    return {'content': selected, 'audit': audit}
