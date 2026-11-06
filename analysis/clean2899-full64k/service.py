import os,sys,json
from pathlib import Path
P=Path('/root/autodl-tmp/Glasseek');E=P/'Evaluation';R=E/'data/runtime';Q=P.parent/'Quest';py=str(Q/'workspace/env/bin/python');name=sys.argv[1]
os.environ.update(NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
secrets=Q/'workspace/strict-reproduction/.secrets/api_key_pool.json'
if name=='serper':
 config=P/'analysis/tool-repair-20261106/search-pool-config.json'
 settings=json.loads(config.read_text()) if config.exists() else {'SERPER_PROXY_CONCURRENCY':'64','SERPER_PROXY_QUEUE_LIMIT':'128','SERPER_PROXY_RPS':'0','SERPER_KEY_CONCURRENCY':'1','SERPER_KEY_RPS':'.5'}
 for key,value in settings.items():
  if key.startswith('SERPER_'):os.environ.setdefault(key,str(value))
 os.environ.setdefault('SERPER_KEY_STATE_FILE',str(P/'analysis/tool-repair-20261106/search-key-state.json'))
 os.environ['SERPER_KEY_POOL_FILE']=str(secrets);os.chdir(E/'services');args=[py,'-m','uvicorn','serper_proxy:app','--host','127.0.0.1','--port','18083']
elif name=='fetch':
 keys=json.loads(secrets.read_text());os.environ['JINA_API_KEYS']=','.join(keys['jina']);os.environ['HTTPS_PROXY']='http://127.0.0.1:18200';os.chdir(R/'services');args=[py,'-m','uvicorn','jina_fetch:app','--host','127.0.0.1','--port','18081']
elif name=='bm25':args=[py,str(Q/'inference/bm25_pool_service.py'),'--workers','4','--queue-limit','16','--port','18215']
elif name=='embedding':args=[py,str(Q/'inference/retrieval_vllm_adapter.py'),'--role','embedding','--model','QWEN3-EMBEDDING-8B','--upstream','http://127.0.0.1:18218','--port','18213']
elif name=='bridge':
 os.environ.update(VISIT_BM25_POOL_URL='http://127.0.0.1:18215',VISIT_EMBEDDING_URL='http://127.0.0.1:18213',VISIT_RETRIEVAL_CANDIDATES='64',VISIT_RETRIEVAL_TOP_K='16',GLASSEEK_RETRIEVAL_CONCURRENCY='4');os.chdir(E/'services');args=[py,'-m','uvicorn','quest_retrieval_bridge:app','--host','127.0.0.1','--port','18217']
else:raise ValueError(name)
os.execv(args[0],args)
