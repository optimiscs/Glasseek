"""Loopback-only web command gateway; credentials never enter CLM containers."""
import sys
from pathlib import Path
import asyncio
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'RL_training'),str(ROOT/'Shared')]
from fastapi import FastAPI,HTTPException
from glasseek_training.web_tools import WebTools
app=FastAPI();sessions={};locks={}
@app.get('/health')
def health():return {'status':'ok'}
@app.post('/session/{session}')
async def invoke(session:str, call:dict):
    if call.get('name') not in ('web_search','web_extract'):raise HTTPException(400,'unsupported web command')
    if session not in sessions:sessions[session]=WebTools();locks[session]=asyncio.Lock()
    async with locks[session]:
        tools=sessions[session];tools.begin_turn()
        return {'output':await tools.execute(call)}
