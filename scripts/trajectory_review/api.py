"""Mounted inside the existing local Quest trajectory viewer."""
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from trajectory_review.core import RUBRIC, RUBRIC_HASH, VERSION, connect, submit
from trajectory_review.source_view import blocks as source_blocks

ROOT=Path(__file__).resolve().parents[2]
router=APIRouter(prefix='/api/curation')


def database():
    path=Path(os.environ.get('GLASSEEK_REVIEW_DB',ROOT/'data/processed/trajectory-human-review-v1/reviews.sqlite'))
    if not path.is_file():raise HTTPException(503,'人工审核批次尚未准备完成')
    return connect(path)


@router.get('/cases')
def cases():
    db=database()
    try:
        rows=db.execute('SELECT c.id,c.source,c.payload,c.source_location,(SELECT COUNT(*) FROM reviews r WHERE r.case_id=c.id) AS reviewed FROM cases c ORDER BY c.source,c.id').fetchall()
        return {'rubric':RUBRIC,'rubric_hash':RUBRIC_HASH,'version':VERSION,'cases':[
            {'id':r['id'],'source':r['source'],'question':json.loads(r['payload'])['question'],
             'original_id':json.loads(r['source_location']).get('original_id',''),
             'messages':len(json.loads(r['payload'])['messages']),'reviewed':r['reviewed']} for r in rows]}
    finally:db.close()


@router.get('/cases/{ident}')
def case(ident:str):
    db=database()
    try:
        row=db.execute('SELECT * FROM cases WHERE id=?',(ident,)).fetchone()
        if not row:raise HTTPException(404,'题目不存在')
        data=json.loads(row['payload'])
        data['messages']=[{**m,'content':m['content'][:300],'characters':len(m['content'])} for m in data['messages']]
        has_files=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_files'").fetchone()
        files=[{'kind':r['kind'],**json.loads(r['metadata'])} for r in db.execute('SELECT kind,metadata FROM case_files WHERE case_id=? ORDER BY kind',(ident,))] if has_files else []
        return {**data,'files':files,'content_hash':row['content_hash'],'policy_hash':row['policy_hash'],'rubric_hash':RUBRIC_HASH,
                'source_location':json.loads(row['source_location'])}
    finally:db.close()


@router.get('/cases/{ident}/files/{kind}')
def source_file(ident:str,kind:str,offset:int=0,download:bool=False):
    if kind not in ('candidate','dataset') or offset<0:raise HTTPException(400,'无效文件类型或偏移')
    db=database()
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_files'").fetchone():
            raise HTTPException(404,'此批次尚未准备原始记录')
        row=db.execute('SELECT metadata,content FROM case_files WHERE case_id=? AND kind=?',(ident,kind)).fetchone()
        if not row:raise HTTPException(404,'原始记录不可用')
        meta=json.loads(row['metadata']);text=row['content']
        if download:
            return Response(text,media_type='application/json',headers={
                'Content-Disposition':f'attachment; filename="{ident}-{kind}.json"','Cache-Control':'no-store'})
        end=min(len(text),offset+30000)
        return {'metadata':meta,'content':text[offset:end],'next_offset':end if end<len(text) else None,'total':len(text)}
    finally:db.close()


@router.get('/cases/{ident}/messages/{index}')
def message(ident:str,index:int,offset:int=0):
    if offset<0:raise HTTPException(400,'无效偏移')
    db=database()
    try:
        row=db.execute('SELECT payload FROM cases WHERE id=?',(ident,)).fetchone()
        if not row:raise HTTPException(404,'题目不存在')
        messages=json.loads(row['payload'])['messages']
        if not 0<=index<len(messages):raise HTTPException(404,'消息不存在')
        text=messages[index]['content'];end=min(len(text),offset+30000)
        return {'content':text[offset:end],'next_offset':end if end<len(text) else None,'total':len(text)}
    finally:db.close()


@router.get('/cases/{ident}/files/{kind}/readable')
def readable_source(ident:str,kind:str):
    raw=source_file(ident,kind,download=True)
    blocks=source_blocks(raw.body.decode())
    return {'blocks':[{**b,'content':b['content'][:300],'characters':len(b['content'])} for b in blocks]}


@router.get('/cases/{ident}/files/{kind}/readable/{index}')
def readable_block(ident:str,kind:str,index:int,offset:int=0):
    if offset<0:raise HTTPException(400,'无效偏移')
    raw=source_file(ident,kind,download=True)
    blocks=source_blocks(raw.body.decode())
    if not 0<=index<len(blocks):raise HTTPException(404,'原始消息不存在')
    text=blocks[index]['content'];end=min(len(text),offset+30000)
    return {'content':text[offset:end],'next_offset':end if end<len(text) else None,'total':len(text)}


def search_query(payload):
    query=payload.get('query')
    if not isinstance(query,str) or not 1<=len(query)<=256:
        raise HTTPException(422,'搜索词长度须为 1–256 个字符')
    return query


def text_hits(text,pattern,limit=2000):
    spans=[];total=0
    for match in pattern.finditer(text):
        total+=1
        if len(spans)<limit:spans.append(match.span())
    # Browser string offsets use UTF-16, including non-BMP characters.
    boundaries={p for span in spans for p in span};positions={};units=0
    if boundaries:
        last=max(boundaries)
        for point,char in enumerate(text):
            if point in boundaries:positions[point]=units
            if point>=last:break
            units+=2 if ord(char)>0xffff else 1
        if len(text) in boundaries:positions[len(text)]=units
    return [{'start':positions[a],'end':positions[b]} for a,b in spans],total


@router.post('/cases/{ident}/search')
def search_messages(ident:str,payload:dict):
    query=search_query(payload)
    db=database()
    try:
        row=db.execute('SELECT payload FROM cases WHERE id=?',(ident,)).fetchone()
        if not row:raise HTTPException(404,'题目不存在')
        pattern=re.compile(re.escape(query),re.IGNORECASE)
        hits=[];total=0;message_count=0
        for index,m in enumerate(json.loads(row['payload'])['messages']):
            found,count=text_hits(m['content'],pattern,2000-len(hits))
            total+=count;message_count+=bool(count)
            hits.extend({'message_index':index,**hit} for hit in found)
        return {'query':query,'hits':hits,'total':total,'message_count':message_count,'limited':total>len(hits)}
    finally:db.close()


@router.post('/cases/{ident}/files/{kind}/search')
def search_source_file(ident:str,kind:str,payload:dict):
    query=search_query(payload)
    if kind not in ('candidate','dataset'):raise HTTPException(400,'无效文件类型')
    db=database()
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='case_files'").fetchone():
            raise HTTPException(404,'此批次尚未准备原始记录')
        row=db.execute('SELECT content FROM case_files WHERE case_id=? AND kind=?',(ident,kind)).fetchone()
        if not row:raise HTTPException(404,'原始记录不可用')
        view=payload.get('view','json')
        if view not in ('json','messages'):raise HTTPException(422,'无效展示模式')
        pattern=re.compile(re.escape(query),re.IGNORECASE)
        if view=='messages':
            hits=[];total=0
            for block in source_blocks(row['content']):
                found,count=text_hits(block['content'],pattern,2000-len(hits));total+=count
                hits.extend({'block_index':block['index'],**hit} for hit in found)
        else:hits,total=text_hits(row['content'],pattern)
        return {'query':query,'hits':hits,'total':total,'limited':total>len(hits)}
    finally:db.close()


@router.post('/reviews')
async def review(request:Request):
    origin=request.headers.get('origin')
    if request.headers.get('x-curation-review')!='1' or (origin and urlsplit(origin).netloc!=request.headers.get('host')):
        raise HTTPException(403,'仅允许查看器同源提交')
    raw=await request.body()
    if len(raw)>65536:raise HTTPException(413,'提交内容过大')
    db=database()
    try:
        try:result=submit(db,json.loads(raw))
        except (ValueError,TypeError,KeyError) as exc:raise HTTPException(422,str(exc)) from None
        return result
    finally:db.close()


@router.get('/summary')
def summary():
    db=database()
    try:
        events=[json.loads(r[0]) for r in db.execute('SELECT event FROM reviews ORDER BY sequence')]
        latest={}
        for e in events:latest[(e['case_id'],e['reviewer'])]=e
        records=list(latest.values())
        counts={k:sum(e['comparison'][k] for e in records) for k in ('agree','disagree','uncomparable')}
        return {'submissions':len(events),'latest_reviews':len(records),'reviewed_cases':len({e['case_id'] for e in events}),
                'cases':db.execute('SELECT COUNT(*) FROM cases').fetchone()[0],**counts,
                'training_ready':False,'gold_status':'single_review_unadjudicated',
                'sampling':'定向抽检；一致率不能代表总体质量。重复审核按每人每题最新版本统计。'}
    finally:db.close()


@router.get('/export')
def export():
    db=database()
    try:
        text='\n'.join(r[0] for r in db.execute('SELECT event FROM reviews ORDER BY sequence'))
        return Response(text+('\n' if text else ''),media_type='application/x-ndjson',
            headers={'Content-Disposition':'attachment; filename="human-review-events.jsonl"','Cache-Control':'no-store'})
    finally:db.close()
