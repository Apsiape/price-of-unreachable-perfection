"""Finite checks of Theorem 6.5 and the depth interpolation in Theorem 6.6.
Standard library only; one process. These are examples, not asymptotic proofs.
"""
import json, math, random, sys
from fractions import Fraction as F
rng=random.Random(20261028)
cases=0
for a in (1,2,3):
 for lam in (F(1,4),F(1,2),F(3,4)):
  for eps in (F(1,10),F(1,1000),F(1,1000000)):
   for trial in range(40):
    S=F(1);cost=0;stopped=False
    for depth in range(1000):
     L=rng.randrange(1,1000)
     if S*6/F(L**a)<=eps:stopped=True;break
     cost+=L
     S*=lam*F(rng.randrange(0,11),10)
    assert stopped
    bound=float(6/eps)**(1/a)/(1-float(lam)**(1/a))
    assert cost<=bound*(1+1e-12)
    cases+=1
depths=[]
for C in (1,2,8,64):
 vals=[(k+1)**2-k*math.log2(C) for k in (100,1000,10000)]
 ratios=[((k+2)**2-(k+1)*math.log2(C))/x for k,x in zip((100,1000,10000),vals)]
 assert abs(ratios[-1]-1)<0.001
 # The loss ratios, unlike the logarithmic depth ratios, approach zero.
 assert C*2.**(-(2*100+3))<1e-50
 depths.append(dict(C=C,depth_ratios=ratios))
result=dict(PASS=True,stopping_cases=cases,precision_depths=depths)
print(json.dumps(result,indent=2))
if len(sys.argv)>1:
 with open(sys.argv[1],'w',encoding='utf8',newline='\n') as out:json.dump(result,out,indent=2);out.write('\n')
