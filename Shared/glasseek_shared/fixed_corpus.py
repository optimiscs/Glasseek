"""Explicit fixed-corpus tool routing; never falls back to the live web."""
import json
import os
from pathlib import Path
import urllib.request


def execute(call, max_chars=20000, timeout=60):
    config=json.loads(Path(os.environ['GLASSEEK_FIXED_CORPUS_CONFIG']).read_text())
    name,args=call['name'],call['arguments']
    body={'opaque_id':os.environ.get('BROWSE_COMP_RUN_ID','').rsplit('__r',1)[0],
          'namespace':os.environ['GLASSEEK_CORPUS_RUN']}
    if name=='web_search':
        body.update(q=args['query'],num=args.get('limit',10));endpoint=config['search_endpoint']
    elif name=='web_extract':
        body['url']=args['url'];endpoint=config['fetch_endpoint'].rstrip('/')+'/fetch'
    else:raise ValueError('fixed corpus only supports web_search and web_extract')
    request=urllib.request.Request(endpoint,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=timeout) as response:result=json.load(response)
    if result.get('success') is False:raise ValueError(result.get('error','fixed corpus request failed'))
    return json.dumps(result,ensure_ascii=False) if name=='web_search' else result['data'][:max_chars]
