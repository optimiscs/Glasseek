import torch,sys
from pathlib import Path
p=Path('experiments/32k-parallel-benchmark')
for case in sys.argv[1:]:
 for typ in ['weights-','']:
  dot=aa=bb=diff=0.; count=0; unequal=0
  for r in range(4):
   a=torch.load(p/f'verified-grads-A/{typ}rank-{r}.pt',map_location='cpu',weights_only=True); b=torch.load(p/f'verified-grads-{case}/{typ}rank-{r}.pt',map_location='cpu',weights_only=True)
   assert a.keys()==b.keys()
   for k in a:
    x=a[k].flatten().double();y=b[k].flatten().double(); assert x.shape==y.shape
    unequal+=(x!=y).sum().item(); dot+=(x@y).item(); aa+=(x@x).item(); bb+=(y@y).item(); diff+=((x-y)@(x-y)).item(); count+=x.numel()
  print(case,typ or 'gradients','count',count,'unequal',unequal,'cosine',dot/(aa*bb)**.5 if aa*bb else None,'relative_l2',(diff/aa)**.5 if aa else None,'norm_ratio',(bb/aa)**.5 if aa else None,flush=True)
