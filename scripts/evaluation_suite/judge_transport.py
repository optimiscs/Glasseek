"""Explicit OpenAI-compatible judge transport; author rubrics/metrics stay intact.

JSON object transport validates the exact author's schema locally. No image
stripping, screenshot transformation, answer repair, or silent provider fallback.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys


def credentials(cfg):
    key=os.environ.get(cfg.get('key_env','OPENAI_API_KEY'))
    if not key and cfg.get('credential_file'):
        data=json.loads(Path(cfg['credential_file']).read_text())
        for part in cfg.get('credential_field','common.JUDGE_OPENAI_API_KEY').split('.'):data=data[part]
        key=data
    if not key:raise RuntimeError('judge credential missing (key values are never logged)')
    os.environ['OPENAI_API_KEY']=key
    if cfg.get('base_url'):os.environ['OPENAI_BASE_URL']=cfg['base_url']
    return key


def adapt(kwargs,cfg):
    out=dict(kwargs);out['model']=cfg['model']
    if 'max_completion_tokens' in out:out['max_tokens']=out.pop('max_completion_tokens')
    out.pop('reasoning_effort',None)
    schema=out.pop('response_format',None);validator=None
    if isinstance(schema,type) and hasattr(schema,'model_json_schema'):
        spec=schema.model_json_schema();validator=schema.model_validate_json
    elif isinstance(schema,dict) and schema.get('type')=='json_schema':
        spec=schema['json_schema']['schema']
        def validator(text):
            import jsonschema
            value=json.loads(text)
            try:jsonschema.validate(value,spec)
            except jsonschema.ValidationError as exc:
                raise ValueError('JSON schema violation at '+'.'.join(map(str,exc.path))) from exc
            return value
    else:spec=None
    if spec is not None:
        out['response_format']={'type':'json_object'}
        out['messages']=[{'role':'system','content':'Return a JSON object matching this schema: '+json.dumps(spec,ensure_ascii=False)}]+list(out['messages'])
    elif schema is not None:out['response_format']=schema
    if cfg.get('text_only',False):
        for m in out.get('messages',[]):
            if isinstance(m.get('content'),list) and any(b.get('type') in ('image_url','input_audio','video') for b in m['content'] if isinstance(b,dict)):
                raise ValueError('configured judge lacks required vision/audio modality; evidence retained, no score substituted')
    return out,validator


def install(cfg):
    if cfg.get('transport')!='json_object_local_schema':return
    from openai.resources.chat.completions import Completions,AsyncCompletions
    if getattr(Completions,'_glasseek_suite_transport',False):return
    sync,async_=Completions.create,AsyncCompletions.create
    def record(result):
        if not cfg.get('usage_log'):return
        path=Path(cfg['usage_log']);path.parent.mkdir(parents=True,exist_ok=True)
        usage=getattr(result,'usage',None)
        with path.open('a') as f:f.write(json.dumps({'model':getattr(result,'model',cfg['model']),'usage':usage.model_dump() if usage else None})+'\n')
        # Persist response BEFORE schema validation so failures remain auditable.
        # Never save request headers, credentials, or client objects.
        with path.with_name('api_responses.jsonl').open('a') as f:
            f.write(json.dumps({'id':getattr(result,'id',None),
                'model':getattr(result,'model',cfg['model']),
                'choices':[{'finish_reason':c.finish_reason,'content':c.message.content,
                            'reasoning_content':getattr(c.message,'reasoning_content',None)} for c in result.choices],
                'usage':usage.model_dump() if usage else None},ensure_ascii=False)+'\n')
    def validate(result,validator):
        record(result)
        if not validator:return None
        if result.choices[0].finish_reason=='length':raise ValueError('structured judge output truncated')
        return validator(result.choices[0].message.content)
    def correction(kw,result,exc):
        if cfg.get('usage_log'):
            with Path(cfg['usage_log']).with_name('schema_errors.jsonl').open('a') as f:
                f.write(json.dumps({'response_id':getattr(result,'id',None),'error_type':type(exc).__name__})+'\n')
        return {**kw,'messages':[*kw['messages'],
            {'role':'assistant','content':result.choices[0].message.content or ''},
            {'role':'user','content':'The previous output did not satisfy the required JSON schema. Return a complete valid object using exactly that schema and the original evaluation criteria. Include every required field with its required type. Do not change the rubric or infer a passing grade from this correction request.'}]}
    def request(self,args,kwargs,parsed=False):
        kw,validator=adapt(kwargs,cfg)
        if parsed and not validator:raise ValueError('native structured schema missing')
        for attempt in range(2):
            result=sync(self,*args,**kw)
            try:value=validate(result,validator)
            except (ValueError,TypeError) as exc:
                if not validator:raise
                kw=correction(kw,result,exc)
                if attempt:raise
                continue
            if parsed:object.__setattr__(result.choices[0].message,'parsed',value)
            return result
    async def arequest(self,args,kwargs,parsed=False):
        kw,validator=adapt(kwargs,cfg)
        if parsed and not validator:raise ValueError('native structured schema missing')
        for attempt in range(2):
            result=await async_(self,*args,**kw)
            try:value=validate(result,validator)
            except (ValueError,TypeError) as exc:
                if not validator:raise
                kw=correction(kw,result,exc)
                if attempt:raise
                continue
            if parsed:object.__setattr__(result.choices[0].message,'parsed',value)
            return result
    def create(self,*args,**kwargs):return request(self,args,kwargs)
    async def acreate(self,*args,**kwargs):return await arequest(self,args,kwargs)
    def parse(self,*args,**kwargs):return request(self,args,kwargs,True)
    async def aparse(self,*args,**kwargs):return await arequest(self,args,kwargs,True)
    Completions.create=create;AsyncCompletions.create=acreate
    Completions.parse=parse;AsyncCompletions.parse=aparse;Completions._glasseek_suite_transport=True
    if cfg.get('responses_as_chat'):
        from openai.resources.responses import Responses
        from types import SimpleNamespace
        def responses_create(self,**kwargs):
            if not isinstance(kwargs.get('input'),str) or set(kwargs)-{'model','input','instructions','max_output_tokens','reasoning'}:
                raise ValueError('unsupported native Responses request; no fields silently removed')
            messages=[]
            if kwargs.get('instructions'):messages.append({'role':'system','content':kwargs['instructions']})
            messages.append({'role':'user','content':kwargs['input']})
            result=self._client.chat.completions.create(model=cfg['model'],messages=messages,
                max_tokens=kwargs.get('max_output_tokens',4096))
            return SimpleNamespace(output_text=result.choices[0].message.content,usage=result.usage)
        Responses.create=responses_create


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--benchmark',required=True);p.add_argument('--script',type=Path,required=True);p.add_argument('--usage-log')
    a,extra=p.parse_known_args();cfg=json.loads(a.config.read_text())[a.benchmark]
    if a.usage_log:cfg['usage_log']=a.usage_log
    credentials(cfg);install(cfg)
    sys.dont_write_bytecode=True;sys.path.insert(0,str(a.script.parent));sys.argv=[str(a.script)]+extra
    runpy.run_path(str(a.script),run_name='__main__')
