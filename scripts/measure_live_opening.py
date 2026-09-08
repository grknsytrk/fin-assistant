import json,time
from pathlib import Path
import httpx
base='https://grknsytrk-fin-api.hf.space'
paths=['/kap/snapshot?company=BIMAS&max_quarters=5','/kap/snapshot?company=BIMAS&max_quarters=20','/kap/price?symbol=BIMAS','/market/stocks/cards/chart?symbol=BIMAS&range=1d','/funds','/funds/categories','/funds/AAL','/funds/AAL/performance?start_date=2026-03-07','/funds/AAL/yield-summary','/funds/AAL/allocations','/funds/AAL/holdings']
results=[]
with httpx.Client(timeout=22,follow_redirects=True) as client:
 for repeat in (1,2):
  for path in paths:
   start=time.perf_counter()
   try:
    response=client.get(base+path)
    elapsed=round((time.perf_counter()-start)*1000)
    data=response.json(); meta=data.get('source_metadata') or {}
    row={'repeat':repeat,'path':path,'ms':elapsed,'status':response.status_code,'decoded_bytes':len(response.content),'wire_bytes':response.headers.get('content-length'),'encoding':response.headers.get('content-encoding'),'cache':data.get('response_cache_status') or data.get('cache_status'),'pending':data.get('pending') or data.get('refresh_pending') or meta.get('refresh_pending'),'rows':len(data.get('rows') or data.get('quarters') or data.get('points') or []),'job_status':(meta.get('history_job') or {}).get('status')}
   except Exception as exc:
    row={'repeat':repeat,'path':path,'ms':round((time.perf_counter()-start)*1000),'error':type(exc).__name__}
   results.append(row);print(json.dumps(row),flush=True)
Path('docs/live-api-timing-2026-09-07.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
