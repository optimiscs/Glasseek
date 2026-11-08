"""Bounded paid smoke test: four unique queries and one idempotent replay, no client retries."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import httpx
import redis

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Evaluation/services'))
from search_pool import load_config


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Exclusive file creation prevents accidentally rerunning an already recorded paid probe.
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as output:
        config=load_config();prefix='{'+config['SEARCH_POOL_NAMESPACE']+'}:'
        state=redis.Redis.from_url(config['SEARCH_POOL_REDIS_URL'],decode_responses=True)
        previous=state.xrevrange(prefix+'events',count=1)
        cursor=previous[0][0] if previous else '0-0'
        run=uuid.uuid4().hex
        queries=['Python asyncio official documentation','Redis scripting Lua official documentation',
                 'FastAPI lifespan official documentation','aiohttp ClientSession official documentation']
        calls=[];started=time.monotonic()
        with httpx.Client(trust_env=False,timeout=float(config['SEARCH_SERVER_TIMEOUT'])+2) as client:
            for index in (0,1,2,3,3):
                token=f'{run}-{index}';at=time.monotonic()
                try:
                    response=client.post(config['SEARCH_SERVER_ENDPOINT'],json={'q':queries[index],'num':10},
                                         headers={'Idempotency-Key':token})
                    data=response.json()
                    calls.append(dict(request_id=hashlib.sha256(token.encode()).hexdigest(),
                        status=response.status_code,latency_ms=(time.monotonic()-at)*1000,
                        queue_ms=float(response.headers.get('X-Search-Queue-Ms',0)),
                        result_count=len(data.get('organic',[])),error_code=data.get('error_code')))
                    if response.status_code!=200:break
                except (httpx.HTTPError,ValueError) as exc:
                    calls.append({'request_id':hashlib.sha256(token.encode()).hexdigest(),'transport_error':type(exc).__name__})
                    break
        ids={r['request_id'] for r in calls}
        # Save observations before ledger inspection; a reporting failure must
        # never require another paid probe. Inclusive IDs work with Redis 6.0.
        json.dump({'calls':calls,'ledger_status':'pending'},output,indent=2);output.flush()
        events=[fields for _,fields in state.xrange(prefix+'events',min=cursor) if fields.get('request_id') in ids]
        attempts=[r for r in events if r['event']=='attempt']
        responses=[json.loads(r['details']) for r in events if r['event']=='response']
        credits=sum(r['credits'] for r in responses if isinstance(r.get('credits'),(int,float)))
        replay_verified=(len(calls)==5 and calls[-1]['request_id']==calls[-2]['request_id']
                         and sum(r['request_id']==calls[-1]['request_id'] for r in attempts)==1
                         and calls[-1].get('status')==200)
        report=dict(http_request_limit=5,upstream_attempt_ceiling=12,unique_query_limit=4,
                    calls=calls,elapsed_seconds=time.monotonic()-started,upstream_attempts=len(attempts),
                    confirmed_response_credits=credits,idempotent_replay_verified=replay_verified,
                    billing_unknown=sum(r['event']=='billing_unknown' for r in events),
                    real_agent_trajectories_per_hour=None,model_utilization=None,
                    limitation='Search-only paid smoke test; no model/agent workload. Credits are provider response fields, not an account balance audit.')
        output.seek(0);output.truncate()
        json.dump(report,output,indent=2);output.write('\n');state.close()
        print(json.dumps({k:v for k,v in report.items() if k!='calls'}))
        if not replay_verified or any(r.get('status')!=200 for r in calls):
            raise SystemExit('Real smoke test incomplete; inspect report before any further paid request')


if __name__=='__main__':main()
