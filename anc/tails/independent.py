"""Independent tail-and-scale finite attacks. Single process; no files outside out are written."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import itertools
import math
from fractions import Fraction
import numpy as np


def check_exhaustive_lists():
    """Use direct character sums, not the shipped Walsh transform."""
    total = 0
    for n, max_n in ((1, 8), (2, 7), (3, 5)):
        q = 1 << n
        characters = [[1 - 2 * ((z & u).bit_count() & 1) for z in range(q)] for u in range(1, q)]
        for N in range(1, max_n + 1):
            expected = sum(math.comb(N, r) * math.exp((N-2*r)**2/(4*N)) for r in range(N+1)) / 2**N
            assert expected < math.sqrt(2)
            best = None
            weighted = 0.0
            weight_sum = 0
            for zs in itertools.combinations_with_replacement(range(q), N):
                counts = [zs.count(z) for z in range(q)]
                weight = math.factorial(N)
                for c in counts:
                    weight //= math.factorial(c)
                cs = [sum(row[z] for z in zs)/N for row in characters]
                phi = sum(math.exp(N*c*c/4) for c in cs)/(q-1)
                weighted += weight*phi
                weight_sum += weight
                total += 1
                if best is None or phi < best[0]:
                    best = phi, cs
            assert weight_sum == q**N
            assert abs(weighted/weight_sum-expected) < 1e-11
            phi, cs = best
            assert phi < math.sqrt(2)
            # Every nontrivial threshold of the chosen discrete spectral law,
            # including thresholds immediately below every atom.
            for atom in sorted(set(abs(c) for c in cs)):
                for threshold in (atom, max(0.0, atom-1e-10)):
                    tail = sum(abs(c)>threshold for c in cs)/(q-1)
                    assert tail <= math.sqrt(2)*math.exp(-N*threshold**2/4)+1e-12
            for nu in (Fraction(1,4), Fraction(1,2), 1, 2, 5, 11, 27):
                nu = float(nu)
                moment = sum(abs(c)**(2*nu) for c in cs)/(q-1)
                assert moment <= math.sqrt(2)*(4*nu/(math.e*N))**nu+1e-11
        print(f"exact multiset enumeration: n={n}, N=1..{max_n}, expectation and selected-list tails/moments PASS", flush=True)
    print("multisets checked:", total, flush=True)


def check_full_fock():
    """Full exterior space (not just an irreducible CAR module); repeated/zero labels."""
    N, n = 2, 2
    modes = 3*N
    dim = 1 << modes
    eye = np.eye(dim, dtype=complex)
    gamma = []
    for j in range(modes):
        g = np.zeros((dim, dim), dtype=complex)
        for state in range(dim):
            sign = 1-2*((state & ((1<<j)-1)).bit_count() & 1)
            g[state ^ (1<<j), state] = sign
        gamma.append(g)
    worst = 0.0
    for zs in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 3)):
        vectors = [np.array([1-2*((z & u).bit_count() & 1) for z in zs])/math.sqrt(N) for u in range(1<<n)]
        X, Z, U = [], [], []
        for u in range(1<<n):
            abc = [sum(vectors[u][j]*gamma[3*j+a] for j in range(N)) for a in range(3)]
            X.append(1j*abc[0]@abc[1])
            Z.append(1j*abc[0]@abc[2])
            signs = [1-2*((zs[j//3] & u).bit_count() & 1) for j in range(modes)]
            diagonal = [math.prod(signs[j] for j in range(modes) if state & (1<<j)) for state in range(dim)]
            U.append(np.diag(diagonal))
        for u, v in itertools.product(range(1<<n), repeat=2):
            for defect in (U[u]@U[v]-U[u^v], U[u]@X[v]@U[u].conj().T-X[u^v],
                           U[u]@Z[v]@U[u].conj().T-Z[u^v], (X[u]@Z[u])@(X[u]@Z[u])+eye):
                worst = max(worst, float(np.max(np.abs(defect))))
            c = float(vectors[u] @ vectors[v])
            for a, b, mixed in ((X[u],Z[v],True),(Z[u],X[v],True),(X[u],X[v],False),(Z[u],Z[v],False)):
                R = (a@b@a.conj().T@b.conj().T-eye)/2
                if mixed:
                    worst = max(worst, float(np.max(np.abs(R.conj().T@R-c*c*eye))))
                else:
                    assert np.linalg.norm(R, 2) <= 2*abs(c)*math.sqrt(max(0,1-c*c))+1e-7
    assert worst < 1e-10
    print(f"full exterior-Fock space dim={dim}, 5 degenerate/nondegenerate lists: representation, covariance, marker, lamp residuals PASS; max error={worst:.3g}", flush=True)


def check_count_constructions():
    # Random legal F, exact offset-corrected Corollary 4.2 count.
    rng = np.random.default_rng(180028)
    F = np.r_[17, 17+np.cumsum(rng.integers(0,2,size=20000))]
    k = F[::2]//2-F[0]//2
    assert np.all((np.diff(k)==0)|(np.diff(k)==1))
    assert np.all(k <= np.arange(len(k)))
    assert max(abs(2*k[j//2]-F[j]) for j in range(len(F))) <= int(F[0])+3

    # A concrete wrong example in the claim: nondecreasing g does not make n/g nondecreasing.
    g = lambda j: 1 << ((j.bit_length()-1)//2)
    assert 15//g(15)==7 and 16//g(16)==4
    print("B count conversion PASS; n/g monotonicity counterexample: F(15)=7, F(16)=4", flush=True)

    # Explicit legal O(J)-delay replacement for E: distribute floor(m(r+1))-floor(mr) ones in dyadic epochs.
    for m in (Fraction(3,2),Fraction(7,1),Fraction(101,3),Fraction(200,1)):
        r0 = max(10, math.ceil(math.log2(math.ceil(m)+1)))
        J0 = 1 << r0
        count = 0
        errs = []
        for J in range(J0, 1<<19):
            r = J.bit_length()-1
            allot = (m*(r+1)).numerator//(m*(r+1)).denominator - (m*r).numerator//(m*r).denominator
            count += int(J-(1<<r) < allot)
            errs.append(float(count-m*math.log2(J+1)+m*r0))
        assert min(errs) >= -float(m)-2 and max(errs) <= float(m)+2
    print("E legal dyadic-epoch logarithmic counts for m=3/2,7,101/3,200: bounded O_m(1) discrepancy PASS", flush=True)


def check_d_exact():
    # Event-level exact simulation, including the one pulse entering every dyadic block.
    def run(targets, max_lg):
        J, k, stage, install = 1, 0, 0, 1
        ends, samples = [], []
        while J.bit_length()-1 < max_lg and stage < len(targets):
            a = targets[stage]
            b = J.bit_length()-1
            target = min(J, max(1 << ((a.numerator*b)//a.denominator), b*b))
            boundary = (1 << (b+1))-1
            hold_until = max(install+1, (stage+3)*install)
            if target <= 2*k and k <= 2*target and J >= hold_until:
                ends.append((stage,b,(k.bit_length()-1)/max(1,b),float(a)))
                stage += 1
                install = J
                continue
            # Within a dyadic block the target is constant. Batch until the next
            # potential stage-install event or boundary; one boundary pulse is
            # executed separately, using its new target.
            if J < boundary:
                if k > 2*target:
                    h = boundary-J
                elif k < (target+1)//2:
                    h = min(boundary-J, max(1,(target+1)//2-k))
                else:
                    h = min(boundary-J, max(1,hold_until-J))
                k += min(h,max(0,target-k))
                J += h
            else:
                J += 1
                b = J.bit_length()-1
                target = min(J,max(1 << ((a.numerator*b)//a.denominator),b*b))
                k += int(k<target)
            samples.append((stage,(k.bit_length()-1)/max(1,J.bit_length()-1)))
        return ends,samples
    for name, seq in (
        ("0/1",[Fraction(s%2) for s in range(30)]),
        ("1/1",[Fraction(1) for s in range(30)]),
        ("0/0",[Fraction(0) for s in range(30)]),
        ("0.3/0.8",[Fraction(3,10) if s%2==0 else Fraction(4,5) for s in range(30)]),
    ):
        ends,samples = run(seq,4096)
        assert ends and all(0 <= r <= 1 for _,r in samples)
        print(f"D exact event simulation {name}: completed stages={len(ends)}, last stage ends={ends[-4:]}",flush=True)


def check_tail_penalty_and_wall():
    # Independent extremal distribution: integrate the capped exponential survival
    # law by differences, rather than summing overlapping tail estimates.
    for Jstar in (20,100,500):
        for period in (1,3,11):
            kfun = lambda j: j//period
            pfun = lambda l: l*period
            n=2*Jstar
            ell=kfun(Jstar)+2
            logN=2*ell+math.log2(64*n)
            logbounds=[]
            previous_tail = -math.inf  # tail above 1 is zero; possible atom at 1 goes in band 0
            for l in range(ell+30):
                exponent=logN-2*(l+1)-4
                logtail=min(0.0,0.5-(2.0**exponent)/math.log(2)) if exponent<1000 else -math.inf
                if logtail == -math.inf or logtail == previous_tail:
                    logmass = -math.inf
                elif previous_tail == -math.inf:
                    logmass = logtail
                else:
                    logmass = logtail + math.log2(-math.expm1((previous_tail-logtail)*math.log(2)))
                logbounds.append(logmass-2*pfun(l)-2)
                previous_tail = logtail
            # Bound the remaining small-residual interval by its largest penalty.
            logbounds.append(-2*pfun(ell+30)-2)
            top=max(logbounds)
            logpenalty=top+math.log2(sum(2**(v-top) for v in logbounds))
            assert logpenalty <= -n
    # Exact integer-root inequalities used by the diagonal time wall.
    def icbrt(n):
        lo,hi=0,1 << ((n.bit_length()+2)//3)
        while lo<hi:
            mid=(lo+hi+1)//2
            if mid**3 <= n: lo=mid
            else: hi=mid-1
        return lo
    for b in (64,65,100,1000,10**6,10**18):
        hi=icbrt(b*b)
        assert (hi+2)**3 < b**3
        at=b**3
        low_at=icbrt(at)
        high_at=icbrt(at*at)
        assert low_at==b and high_at==b*b
    print("independent threshold/penalty bounds and exact diagonal-wall catch-up inequalities PASS",flush=True)


if __name__ == "__main__":
    check_exhaustive_lists()
    check_full_fock()
    check_count_constructions()
    check_d_exact()
    check_tail_penalty_and_wall()
    print("ALL INDEPENDENT FINITE CHECKS PASSED",flush=True)
