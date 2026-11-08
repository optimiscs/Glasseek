"""Native Hermes session/event recorder for external agent-loop producers.

Persists actual actions and context boundaries through SessionDB; consumers do not
need to know which agent loop produced them. Missing measurements stay missing.
"""
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from hermes_state import SessionDB


def canonical(message):
    return {k:message.get(k) for k in ('role','content','tool_calls','reasoning_content') if message.get(k) is not None}

class AgentSessionRecorder:
    def __init__(self, profile_dir, snapshot, session_id, model, base_url='', cwd='', tools=None, capacity_tokens=None):
        self.profile=Path(profile_dir);self.profile.mkdir(parents=True,exist_ok=True)
        self.db=SessionDB(self.profile/'state.db');self.snapshot=Path(snapshot)
        self.log=self.profile/'logs/agent.log';self.log.parent.mkdir(parents=True,exist_ok=True)
        self.sid=None;self.root=session_id;self.model=model;self.base_url=base_url;self.cwd=cwd
        self.tools=tools;self.capacity_tokens=capacity_tokens
        self.expected=[];self.api_calls=0
        if self.db.get_session(session_id):
            self.sid=self.db.get_compression_tip(session_id)
            self.expected=[canonical(m) for m in self.db.get_messages_as_conversation(self.sid)]
            self.db.reopen_session(self.sid)

    def event(self, component, text, at=None):
        stamp=datetime.fromtimestamp(at or time.time()).strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
        with self.log.open('a') as stream:stream.write(f'{stamp} INFO [{self.sid}] agent.{component}: {text}\n')

    def append(self, message, *, inherited=False, at=None):
        allowed=('tool_name','tool_calls','tool_call_id','finish_reason','reasoning','reasoning_content','reasoning_details')
        message_id=self.db.append_message(self.sid,message['role'],message.get('content'),timestamp=at,
            inherited_from_previous_session=inherited,**{k:message[k] for k in allowed if k in message})
        self.expected.append(copy.deepcopy(canonical(message)))
        return message_id

    def before_query(self, messages, *, record_request=True):
        current=[canonical(m) for m in messages]
        unchanged_prefix=current[:len(self.expected)]==self.expected
        if self.sid is None or not unchanged_prefix:
            parent=self.sid
            if parent:
                self.db.end_session(parent,'compression');self.publish()
            sid=self.root if self.db.get_session(self.root) is None else self.root+'-'+uuid4().hex[:10]
            self.sid=sid;self.expected=[]
            self.db.create_session(sid,'cli',model=self.model,parent_session_id=parent,cwd=self.cwd,
                system_prompt=next((m.get('content','') for m in messages if m['role']=='system'),''))
            for m in messages:self.append(m,inherited=parent is not None)
        else:
            for m in messages[len(self.expected):]:self.append(m)
        if record_request:
            self.request_id=self.db.record_request_context(self.sid,messages,self.tools,capacity_tokens=self.capacity_tokens,provenance='request')
        self.publish()

    def model_response(self, response, start, end):
        msg=response.choices[0].message.model_dump()
        msg['content']=msg.get('content') or ''
        if msg.get('tool_calls'):msg['tool_calls']=msg['tool_calls'][:1]  # executed action, per upstream policy
        usage=response.usage.model_dump() if response.usage else {}
        prompt=usage.get('prompt_tokens',0) or 0;output=usage.get('completion_tokens',0) or 0
        detail=usage.get('prompt_tokens_details') or {};completion=usage.get('completion_tokens_details') or {}
        cached=max(detail.get('cached_tokens',0) or 0,usage.get('cache_read_input_tokens',0) or 0)
        written=usage.get('cache_creation_input_tokens',0) or 0
        self.db.update_token_counts(self.sid,input_tokens=max(0,prompt-cached-written),output_tokens=output,
            cache_read_tokens=cached,cache_write_tokens=written,reasoning_tokens=completion.get('reasoning_tokens',0) or 0,
            model=self.model,api_call_count=1)
        self.db.complete_request_context(self.request_id,prompt)
        self.api_calls+=1
        self.event('conversation_loop',f'API call #{self.api_calls}: model={self.model} provider=custom in={prompt} out={output} total={prompt+output} latency={end-start:.6f}s cache={cached}/{prompt} ({100*cached/prompt if prompt else 0:.0f}%)',end)
        mid=self.append(msg,at=end)
        self.db.record_message_timing(mid,'model',end-start,started_at=start,ended_at=end)
        self.publish()

    def tool_result(self, pending, observation, start, end):
        calls=(pending or {}).get('tool_calls') or []
        if not calls:return
        call=calls[0];name=call['function']['name'];cid=call['id']
        record={'session_id':self.sid,'tool_call_id':cid,'tool_name':name,'duration_seconds':end-start,
                'outcome':'completed' if observation.result.return_code==0 else 'failed'}
        self.event('tool_executor','glasseek_tool_timing '+json.dumps(record),end)
        mid=self.append({'role':'tool','tool_call_id':cid,'tool_name':name,'content':observation.render()},at=end)
        self.db.record_message_timing(mid,'tool',end-start,started_at=start,ended_at=end)
        self.publish()

    def publish(self):
        data=self.db.get_session(self.sid)
        messages=self.db.get_messages_as_conversation(self.sid)
        session=dict(session_id=self.sid,parent_session_id=data.get('parent_session_id'),model=self.model,
            messages=messages,session_start=datetime.fromtimestamp(data['started_at']).isoformat(),
            last_updated=datetime.fromtimestamp(data['ended_at']).isoformat() if data.get('ended_at') else datetime.now().isoformat(),
            session_end=data.get('ended_at'),producer={'pid':os.getpid(),'start_ticks':Path(f'/proc/{os.getpid()}/stat').read_text().split(') ',1)[1].split()[19]},input_tokens=data.get('input_tokens',0),
            output_tokens=data.get('output_tokens',0),cache_read_tokens=data.get('cache_read_tokens',0),
            cache_write_tokens=data.get('cache_write_tokens',0),api_call_count=data.get('api_call_count',0))
        target=self.snapshot.with_name('session_'+self.sid+'.json');target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(session,ensure_ascii=False));tmp.replace(target)
        return session

    def finish(self, reason='completed'):
        if self.sid:self.db.end_session(self.sid,reason);return self.publish()

    def close(self):self.db.close()
