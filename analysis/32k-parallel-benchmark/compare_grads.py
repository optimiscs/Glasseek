import torch, sys
from pathlib import Path
p=Path('analysis/32k-parallel-benchmark')
for case in sys.argv[1:]:
 dot=aa=bb=diff=0.; count=0
 for r in range(4):
  a=torch.load(p/f'seeded-grads-A/rank-{r}.pt',map_location='cpu',weights_only=True)
  b=torch.load(p/f'seeded-grads-{case}/rank-{r}.pt',map_location='cpu',weights_only=True)
  assert a.keys()==b.keys()
  for k in a:
   x=a[k].flatten().double(); y=b[k].flatten().double()
   assert x.shape==y.shape
   dot+=(x@y).item(); aa+=(x@x).item(); bb+=(y@y).item(); diff+=((x-y)@(x-y)).item(); count+=x.numel()
 print(case, 'elements',count,'cosine',dot/(aa*bb)**.5,'relative_l2',(diff/aa)**.5,'norm_ratio',(bb/aa)**.5,flush=True)
