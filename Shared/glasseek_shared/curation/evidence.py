"""Evidence metadata never manufactures timestamps or independent sources."""
import re
from urllib.parse import urlsplit


def enrich(rows):
    result=[]
    for row in rows:
        urls=sorted(set(re.findall(r'https?://[^\s<>"\\]+',row['text'])))
        result.append({**row,'available_after_step':row['step_id'],'urls':urls,
                       'source_domains':sorted({urlsplit(u).hostname for u in urls if urlsplit(u).hostname}),
                       'observed_at':None,'timestamp_status':'not_verified',
                       'independence':'not_inferred_from_url_count',
                       'offset_space':'observation_view_'+row['view']})
    return result
