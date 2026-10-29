"""Independent refutation probes. Single process; no imports from the supplied checker.
The scalar profile countermodel is explicitly abstract, not a finite-game construction.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import itertools, math, json, pathlib, random, time
from fractions import Fraction
import numpy as np

OUT = pathlib.Path(__file__).resolve().parent
rng = random.Random(141028)
results = {}

def record(name, count, **data):
    results[name] = dict(checks=count, PASS=True, **data)
    print(name, count, 'PASS', flush=True)

def table5():
    cases = []
    total_pairs = 0
    for X, L in ((1,3),(1,5),(2,3),(2,4)):
        games = [('s',x) for x in range(X)] + [('o',x,y) for x in range(X) for y in range(X)]
        proofs = [p for size in (1,2,3) for p in itertools.product(range(L), repeat=size)]
        questions = [(g,p) for g in games for p in proofs]
        counts = dict(equality=0, clause=0, first_input=0, second_input=0, trivial=0)
        losing = 0
        for (g,p), (h,q) in itertools.product(questions, repeat=2):
            total_pairs += 1
            if g == h and p == q:
                counts['equality'] += 1
                continue
            # Match both orientations, exactly as Table 5. All answers are zero.
            kind = None
            for u,v,a,b in ((g,h,p,q),(h,g,q,p)):
                if u[0] != 'o' or len(a) != 1:
                    continue
                i = a[0]
                if u == v and len(b) == 3 and i in b:
                    kind = 'clause'  # pessimistically reject EVERY triple
                    break
                # T=1: embedded first input is index 0, second input index 1.
                if v == ('s',u[1]) and b == (0,0) and i == 0:
                    kind = 'first_input'
                    break
                if v == ('s',u[2]) and b == (0,0) and i == 1:
                    kind = 'second_input'
                    break
            if kind is None:
                counts['trivial'] += 1
            else:
                counts[kind] += 1
                losing += kind == 'clause'
        exact = 2*X*X*(3*L**3-3*L**2+L)
        assert losing == exact
        assert sum(counts.values()) == len(questions)**2
        loss = Fraction(losing, len(questions)**2)
        assert loss <= Fraction(6, X*X*L**3) <= Fraction(12,L**3)
        cases.append(dict(X=X,L=L,alphabet=len(questions),counts=counts,losing_pairs=losing,
                          loss=str(loss),loss_float=float(loss)))
    record('Table5_exhaustive', total_pairs, cases=cases,
           note='All four table rows and reversed orientations; clause rejection deliberately maximal.')

def cap():
    worst = 0.0
    for trial in range(400):
        a = rng.choice((1.,2.,3.,0.4,5.))
        logc = rng.uniform(-2,4)
        x = rng.uniform(2,1000)
        logS = 0.
        levels = []
        K = None
        for k in range(1,2000):
            # Jagged, non-monotone size sequence; no regular-growth assumption.
            logL = rng.uniform(0, min(600,x/a+30))
            if logS+logc-a*logL <= -x:
                K = k
                break
            levels.append(logL)
            logS += -1. + math.log2(rng.uniform(0.001,1.))
        assert K is not None
        if levels:
            top=max(levels)
            logsum=top+math.log2(sum(2.**(u-top) for u in levels))
            logbound=(logc+x)/a-math.log2(1.-2.**(-1./a))
            assert logsum <= logbound+1e-10
            worst=max(worst,2.**(logsum-logbound))
    record('stopping_cap_6_5',400,max_cost_over_bound=worst)


def dimension():
    for _ in range(800):
        D=2**rng.randrange(1,12)
        blocks=rng.randrange(1,8)
        weights=[Fraction(rng.randrange(1,30),1) for i in range(blocks)]
        sw=sum(weights)
        weights=[v/sw for v in weights]
        ranks=[rng.randrange(1,3*D) for i in range(blocks)]
        ancillary=[rng.randrange(1,7) for i in range(blocks)]
        support=sum(w*min(Fraction(d,D*r),1) for w,d,r in zip(weights,ranks,ancillary))
        delta=1-support
        assert sum(ranks)>=(1-delta)*D
    record('conditional_corner_rank_6_6',800)

def realization():
    # Refute the claimed pointwise cumulative-cost bound with a convergent approximation.
    counter=[]
    for k in (20,100,400):
        def exponent(j): return (1+(-1)**(j+1)/math.sqrt(j))*j*j
        gap=exponent(k-1)-exponent(k)-math.log2(k+1)
        assert gap>0
        counter.append(dict(k=k,log2_N_previous_over_claimed_bound=gap))
    normalized=[]
    for k in (100,400,2000,6000):
        logs=[(1+(-1)**(j+1)/math.sqrt(j))*j*j for j in range(1,k+1)]
        top=max(logs)
        logsum=top+math.log2(sum(2.**(u-top) for u in logs))
        normalized.append(dict(k=k,log2_sum_N_over_k_squared=logsum/(k*k)))
    for k in range(1,100):
        assert sum(2*j+1 for j in range(k))==k*k
        assert sum(2*(j+1) for j in range(k))==k*(k+1)
    record('conditional_schedule_and_cost',202,cumulative_bound_counterexamples=counter,
           normalized_asymptotics=normalized)

def main():
    start=time.time()
    for fn in (table5,cap,dimension,realization): fn()
    results['summary']=dict(PASS=True,groups=len(results),elapsed_seconds=time.time()-start,
                            total_checks=sum(row['checks'] for row in results.values()))
    (OUT/'independent_checks.json').write_text(json.dumps(results,indent=2),encoding='utf8')
    print(json.dumps(results['summary']),flush=True)

if __name__=='__main__': main()
