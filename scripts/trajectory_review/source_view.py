"""Read-only presentation of source messages; never rewrites source snapshots."""
import json


def body(value):
    if isinstance(value,str):
        # Decode only complete JSON string wrappers, never blanket-unescape paths/code.
        for _ in range(2):
            try:decoded=json.loads(value)
            except (ValueError,TypeError):break
            if not isinstance(decoded,str):break
            value=decoded
        return value
    if isinstance(value,list) and all(isinstance(p,dict) and p.get('type')=='text' for p in value):
        return '\n\n'.join(str(p.get('text','')) for p in value)
    return json.dumps(value,ensure_ascii=False,indent=2)


def blocks(text):
    raw=json.loads(text);result=[]
    def add(role,content,path,title):
        result.append(dict(index=len(result),role=str(role),content=body(content),path=path,title=title))
    def walk(record,path='$',group='主轨迹'):
        if not isinstance(record,dict):
            add('metadata',record,path,group);return
        used=set()
        for key in ('system','system_prompt'):
            if isinstance(record.get(key),str) and record[key]:
                add('system',record[key],path+'.'+key,group+' · 系统');used.add(key)
        for key in ('messages','trajectory','conversations'):
            messages=record.get(key)
            if isinstance(messages,str):
                try:messages=json.loads(messages)
                except ValueError:continue
            if not isinstance(messages,list):continue
            used.add(key)
            for i,m in enumerate(messages):
                pointer=f'{path}.{key}[{i}]'
                if not isinstance(m,dict):add('unknown',m,pointer,group+f' · 消息 {i+1}');continue
                role=m.get('role',m.get('from',m.get('speaker','unknown')))
                content_key=next((k for k in ('content','value','text') if k in m),None)
                add(role,m.get(content_key,'') if content_key else '',pointer,group+f' · 消息 {i+1}')
                extra={k:v for k,v in m.items() if k not in ('role','from','speaker',content_key)}
                if extra:add(role,extra,pointer,group+f' · 消息 {i+1} 附加字段')
        children=record.get('subagents')
        if isinstance(children,list):
            used.add('subagents')
            for i,child in enumerate(children):walk(child,f'{path}.subagents[{i}]',f'子代理 {i+1}')
        other={k:v for k,v in record.items() if k not in used}
        if other:add('metadata',other,path,group+' · 其他原始字段')
    walk(raw)
    return result
