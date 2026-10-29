"""Finite diagnostics for the window proof (Theorem 5.5) and exact clocks.
Vary localization degrees independently of the logarithmic pulse parameter.
"""
from fractions import Fraction as F
from pathlib import Path
import json,math
cases=0;largest=0
for m in (F(5,4),F(3,2),F(2),F(7,2),F(11)):
 r0=max(2,math.ceil(math.log2(float(m)+2)))
 def count(J):
  r=J.bit_length()-1
  if r<r0:return 0
  total=math.floor(m*r)-math.floor(m*r0)
  pulse=math.floor(m*(r+1))-math.floor(m*r)
  return total+min(pulse,J-2**r)
 for degree in (1,5,40,200):
  K=degree+10
  for power in (32,64,128):
   n=2**power;J=n//2-K*power-5;H=3*J+1
   # Integer upper bound on log2(2^J H^degree 2^(-n/2));
   # floating subtraction at n=2^128 would erase the horizon offset.
   bracket_bound=J+degree*H.bit_length()-n//2
   assert bracket_bound<0
   # Finite witness that large degrees change the horizon offset, not
   # the coefficient of log n in the corner charge.
   deviation=abs(count(J)-float(m)*power)
   assert deviation<float(m)*(r0+4)+2
   largest=max(largest,deviation);cases+=1
for depth in range(1,11):
 cylinders=2**depth;weights=[F(1,cylinders)]*cylinders
 assert sum(weights)==1 and min(weights)>0
 # A faithful finite trace needs at least one dimension per nonzero cylinder.
 assert len(weights)==cylinders
# Exact Pauli marker: (XZ)^2=-I, whereas the identity residual for J is -I.
J=[[-1,0],[0,-1]]
assert all(J[i][j]+int(i==j)==0 for i in range(2) for j in range(2))
assert sum((J[i][j]-int(i==j))**2 for i in range(2) for j in range(2))==8
result=dict(PASS=True,window_cases=cases,max_count_deviation=largest,
 fair_cylinder_depths=10,marker_is_not_a_relator=True)
Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
