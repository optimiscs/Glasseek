-- All admission and state transitions are atomic and use the Redis clock.
local p = KEYS[1]
local op = ARGV[1]
local a = cjson.decode(ARGV[2])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local q, active, metrics = p..'queue', p..'active', p..'metrics'
local function metric(name, n) redis.call('HINCRBY', metrics, name, n or 1) end
local function event(kind, id, key, extra)
  redis.call('XADD', p..'events', '*', 'event', kind, 'request_id', id or '',
    'key_id', key or '', 'at', tostring(now), 'details', cjson.encode(extra or {}))
end
local function finish(id, status, body, hold_key)
  local j=p..'job:'..id
  local key=redis.call('HGET',j,'key_id')
  -- An ambiguous HTTP outcome may still be executing upstream. Hold its key slot
  -- until lease expiry instead of immediately admitting another paid request.
  if key and not hold_key then redis.call('ZREM',p..'inflight:'..key,id) end
  redis.call('ZREM',active,id)
  redis.call('ZREM',q,id)
  redis.call('HSET',j,'phase','done','status',status,'body',body,'finished_at',now)
  redis.call('EXPIRE',j,tonumber(redis.call('HGET',p..'config','ttl')))
  metric('completed'); if tonumber(status)==200 then metric('success') end
  event('completed',id,key,{status=status})
end
local function errorbody(code, message, retryable)
  local provider = redis.call('HGET', p..'config', 'provider') or 'search'
  code = string.gsub(code, '^search_', provider..'_')
  return cjson.encode({success=false,error_code=code,error=message,retryable=retryable or false})
end
local function owns(j)
  return redis.call('HGET',j,'owner')==a.owner and
    tonumber(redis.call('ZSCORE',active,a.id) or '0')>now
end
if op=='init' then
  local digest=redis.call('HGET',p..'config','digest')
  if digest and digest~=a.digest then return redis.error_reply('Search namespace configuration mismatch') end
  if not digest then
    for k,v in pairs(a.config) do redis.call('HSET',p..'config',k,v) end
    redis.call('HSET',p..'config','digest',a.digest)
    for k,v in pairs(a.keys) do redis.call('HSET',p..'keys',k,v) end
    for k,v in pairs(a.state) do redis.call('HSET',p..'state',k,cjson.encode(v)) end
    for k,v in pairs(a.usage) do
      for name,n in pairs(v) do redis.call('HSET',p..'usage:'..k,name,n) end
    end
    event('initialized','','',{})
  end
  return 1
elseif op=='sync_keys' then
  redis.call('DEL',p..'keys')
  for k,v in pairs(a.keys) do redis.call('HSET',p..'keys',k,v) end
  event('keys_updated','','',{count=redis.call('HLEN',p..'keys')}); return 1
elseif op=='configure' then
  if redis.call('ZCARD',q)+redis.call('ZCARD',active)>0 then
    return redis.error_reply('Drain the search queue before changing configuration')
  end
  for _,key in ipairs(redis.call('HKEYS',p..'keys')) do
    if redis.call('ZCOUNT',p..'inflight:'..key,now,'+inf')>0 then
      return redis.error_reply('Wait for outstanding credential leases before changing configuration')
    end
  end
  for k,v in pairs(a.config) do redis.call('HSET',p..'config',k,v) end
  redis.call('HSET',p..'config','digest',a.digest)
  event('configured','','',{}); return 1
elseif op=='restore_key' then
  if redis.call('HEXISTS',p..'keys',a.key)==0 then return 0 end
  redis.call('HDEL',p..'state',a.key); event('key_restored','',a.key,{}); return 1
elseif op=='enqueue' then
  local j=p..'job:'..a.id
  if redis.call('EXISTS',j)==1 then
    if redis.call('HGET',j,'digest')~=a.digest then return 'conflict' end
    metric('deduplicated'); return 'existing'
  end
  local capacity=tonumber(redis.call('HGET',p..'config','concurrency'))+tonumber(redis.call('HGET',p..'config','queue'))
  if redis.call('ZCARD',q)+redis.call('ZCARD',active)>=capacity then metric('queue_full'); return 'full' end
  redis.call('HSET',j,'phase','queued','payload',a.payload,'digest',a.digest,'enqueued_at',now)
  redis.call('ZADD',q,now,a.id); metric('accepted'); return 'new'
elseif op=='claim' then
  if redis.call('ZCARD',active)>=tonumber(redis.call('HGET',p..'config','concurrency')) then return {} end
  local ids=redis.call('ZRANGE',q,0,0)
  if #ids==0 then return {} end
  local id=ids[1]; local j=p..'job:'..id
  redis.call('ZREM',q,id)
  if now-tonumber(redis.call('HGET',j,'enqueued_at'))>tonumber(redis.call('HGET',p..'config','queue_timeout')) then
    finish(id,503,errorbody('search_queue_timeout','Timed out waiting for search capacity.',true)); return {}
  end
  redis.call('HSET',j,'owner',a.owner,'phase','running','started_at',now)
  redis.call('ZADD',active,now+a.lease,id)
  return {id,redis.call('HGET',j,'payload'),tostring(now-tonumber(redis.call('HGET',j,'enqueued_at')))}
elseif op=='renew' then
  local j=p..'job:'..a.id
  if not owns(j) then return 0 end
  redis.call('ZADD',active,now+a.lease,a.id)
  local key=redis.call('HGET',j,'key_id')
  if key then redis.call('ZADD',p..'inflight:'..key,now+a.lease,a.id) end
  return 1
elseif op=='admit' then
  local j=p..'job:'..a.id
  if not owns(j) or redis.call('HGET',j,'phase')~='running' then return {'lost'} end
  local candidates=redis.call('HKEYS',p..'keys')
  table.sort(candidates)
  local cursor=tonumber(redis.call('GET',p..'cursor') or '0')
  local eligible=false
  local c=p..'config'
  local limit=tonumber(redis.call('HGET',c,'attempt_limit'))
  if limit>0 and tonumber(redis.call('HGET',metrics,'upstream_attempts') or '0')>=limit then return {'budget'} end
  for i=1,#candidates do
    local index=(cursor+i-1)%#candidates+1; local key=candidates[index]
    local state=redis.call('HGET',p..'state',key)
    local blocked=false
    if state then
      state=cjson.decode(state)
      blocked=state.reason=='quota_exhausted' or state.reason=='invalid_key' or state.reason=='retired' or tonumber(state['until'] or 0)>now
    end
    if not blocked and not a.excluded[key] then
      eligible=true
      local z=p..'inflight:'..key
      redis.call('ZREMRANGEBYSCORE',z,'-inf',now)
      local knext=tonumber(redis.call('HGET',p..'next',key) or '0')
      local gnext=tonumber(redis.call('GET',p..'global_next') or '0')
      if redis.call('ZCARD',z)<tonumber(redis.call('HGET',c,'key_concurrency')) and knext<=now and gnext<=now then
        redis.call('ZADD',z,now+a.lease,a.id)
        redis.call('HSET',p..'next',key,now+1/tonumber(redis.call('HGET',c,'key_rps')))
        local rps=tonumber(redis.call('HGET',c,'rps'))
        if rps>0 then redis.call('SET',p..'global_next',now+1/rps) end
        redis.call('SET',p..'cursor',index)
        redis.call('HSET',j,'phase','sent','key_id',key,'sent_at',now)
        local wait=now-tonumber(redis.call('HGET',j,'enqueued_at'))
        redis.call('HSET',j,'queue_ms',wait*1000)
        redis.call('LPUSH',p..'queue_ms',tostring(wait*1000)); redis.call('LTRIM',p..'queue_ms',0,9999)
        redis.call('HINCRBY',p..'usage:'..key,'attempts',1); metric('upstream_attempts')
        event('attempt',a.id,key,{attempt=a.attempt})
        return {'ok',key,redis.call('HGET',p..'keys',key)}
      end
    end
  end
  if eligible then return {'wait'} else return {'unavailable'} end
elseif op=='response' then
  local j=p..'job:'..a.id
  if not owns(j) then return 0 end
  local key=redis.call('HGET',j,'key_id')
  redis.call('ZREM',p..'inflight:'..key,a.id)
  redis.call('HINCRBY',p..'usage:'..key,'responses',1)
  redis.call('HINCRBY',p..'usage:'..key,'http_'..a.status,1)
  if type(a.credits)=='number' then redis.call('HINCRBYFLOAT',p..'usage:'..key,'confirmed_credits',a.credits)
  else redis.call('HINCRBY',p..'usage:'..key,'responses_without_credit_field',1) end
  metric('upstream_http_'..a.status)
  if a.reason~='' then
    redis.call('HSET',p..'state',key,cjson.encode({reason=a.reason,['until']=now+a.cooldown}))
    metric(a.reason)
  end
  event('response',a.id,key,{status=a.status,credits=a.credits})
  if a.rotate then
    redis.call('HSET',j,'phase','running'); redis.call('HDEL',j,'key_id')
  else finish(a.id,a.result_status,a.body) end
  return 1
elseif op=='finish' then
  local j=p..'job:'..a.id
  if not owns(j) then return 0 end
  if a.unknown then
    metric('billing_unknown'); event('billing_unknown',a.id,redis.call('HGET',j,'key_id'),{})
  end
  finish(a.id,a.status,a.body,a.unknown); return 1
elseif op=='recover' then
  local ids=redis.call('ZRANGEBYSCORE',active,'-inf',now,'LIMIT',0,1024)
  for _,id in ipairs(ids) do
    local j=p..'job:'..id
    if redis.call('HGET',j,'phase')=='sent' then
      metric('billing_unknown'); metric('recovered_uncertain')
      event('billing_unknown',id,redis.call('HGET',j,'key_id'),{worker_lost=true})
      finish(id,503,errorbody('search_outcome_unknown','Worker lost after dispatch; billing may have occurred. Request was not replayed.'))
    else
      redis.call('ZREM',active,id)
      redis.call('HSET',j,'phase','queued'); redis.call('HDEL',j,'owner')
      redis.call('ZADD',q,tonumber(redis.call('HGET',j,'enqueued_at')),id)
      metric('recovered_queued')
    end
  end
  return #ids
else return redis.error_reply('Unknown search state operation') end
