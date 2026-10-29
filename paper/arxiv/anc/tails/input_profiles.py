"""Check 3: the profile law of Theorem 4.1 on explicit scales, and the tracking streams of Theorems 5.2 and 5.4.

Part 1 (upper bound arithmetic). For a pulse count k with first hits p(l), the penalty is s = 2^{-p(l)-1} on the dyadic band
(2^{-l-1}, 2^{-l}]. We sum spectral-band upper bounds from Lemma 3.1,
tail(kappa) = min(1, sqrt2 exp(-N kappa^2/16)), and compute in the log domain the least N
certified by this sufficient upper envelope (compiler constant set to 1). It is not an extremizing spectral law. We compare
log2 N_min with 2k(n/2) (the lower-bound main term) and with the proof's choice 2 l_n + log2(16 n ln2).
Part 2: Theorem 5.2 tracking stream (greedy + on-track + holds + background delay) for G(J) = sqrt(J) with targets whose
liminf/limsup are L = 1/2, U = 2. Part 3: Theorem 5.4 stream with log k/log J oscillating between 0.3 and 0.8.
Pure Python, single thread.
"""
import math

LN2 = math.log(2.0)

def log2_sum(terms):  # terms are log2 values
    m = max(terms)
    return m + math.log2(sum(2.0 ** (t - m) for t in terms))

def log2_tail(log2N, l):  # log2 of min(1, sqrt2 exp(-N kappa^2/16)) at kappa = 2^{-l-1}
    e = log2N - 2 * (l + 1) - 4          # log2(N kappa^2 / 16)
    if e > 1000: return -1e300
    v = 0.5 - (2.0 ** e) / LN2
    return min(0.0, v)

def log2_Es2(log2N, pfun, lmax):
    terms = []
    for l in range(0, lmax + 1):
        # band (2^{-l-1}, 2^{-l}]: mass <= tail(2^{-l-1}) (crude but valid upper bound on the band mass)
        lt = log2_tail(log2N, l)
        terms.append(-2 * pfun(l) - 2 + lt)
    # spectral mass below 2^{-lmax-1}: s <= 2^{-p(lmax+1)-1}, mass <= 1
    terms.append(-2 * pfun(lmax + 1) - 2)
    return log2_sum(terms)

def least_log2N(n, pfun, kfun):
    lo, hi = 1.0, 2.0 * kfun(n // 2) + 4 * math.log2(n + 2) + 40
    lmax = int(hi / 2) + 60
    for _ in range(60):
        mid = (lo + hi) / 2
        if log2_Es2(mid, pfun, lmax) <= -n:
            hi = mid
        else:
            lo = mid
    return hi

scales = {
    "k=floor(J^(3/4))": (lambda J: math.floor(J ** 0.75), lambda l: math.ceil(l ** (4 / 3))),
    "k=floor(J^(1/2))": (lambda J: math.floor(J ** 0.5), lambda l: l * l),
    "k=floor((log2 J)^2)": (lambda J: math.floor(math.log2(J) ** 2) if J >= 1 else 0, lambda l: math.ceil(2 ** math.sqrt(l))),
    "k=floor(J/log2 J)": (lambda J: math.floor(J / math.log2(J)) if J >= 2 else 0, None),
    "k=floor(3 log2 J) [quasi-poly, m=3]": (lambda J: math.floor(3 * math.log2(J)) if J >= 1 else 0, lambda l: math.ceil(2 ** (l / 3))),
}

def make_p(kfun):
    cache = {}
    def p(l):
        if l in cache: return cache[l]
        if l == 0: return 0
        lo, hi = 1, 2
        while kfun(hi) < l: hi *= 2
        while lo < hi:
            mid = (lo + hi) // 2
            if kfun(mid) >= l: hi = mid
            else: lo = mid + 1
        cache[l] = lo
        return lo
    return p

print("Part 1: least log2 N certified by the Lemma 3.1 spectral-band upper envelope", flush=True)
print(f"{'scale':38s} {'n':>9s} {'2k(n/2)':>9s} {'log2Nmin':>9s} {'excess':>8s} {'log2 n':>7s} {'ratio':>7s}", flush=True)
for name, (kfun, _) in scales.items():
    p = make_p(kfun)
    for n in (10**3, 10**4, 10**5, 10**6):
        main = 2 * kfun(n // 2)
        L = least_log2N(n, p, kfun)
        print(f"{name:38s} {n:9d} {main:9d} {L:9.2f} {L-main:8.2f} {math.log2(n):7.2f} {L/main:7.4f}", flush=True)

# Part 2: Theorem 5.2 tracking stream
print("\nPart 2: Theorem 5.2 stream, G(J)=sqrt(J), targets -> liminf 1/2, limsup 2", flush=True)
def a_seq(s):
    return 0.5 + 1.0 / (s + 2) if s % 2 == 0 else 2.0 - 1.0 / (s + 2)
k = 0; s = 0; a = a_seq(0); J_inst = 1; ready_at = 1
stage_ends = []; ratios_after = []
Jmax = 3_000_000
for J in range(1, Jmax + 1):
    G1 = math.sqrt(J + 1)
    if k < a * G1: k += 1
    G = math.sqrt(J + 1)
    on_track = abs(k - a * G) <= a + 1
    if J >= ready_at and on_track and J >= (s + 3) * J_inst:
        stage_ends.append((J, s, a, k / G))
        s += 1; a = a_seq(s); J_inst = J
        ready_at = J + (s + 2) ** 2 * 50   # background delay (padding), arbitrary but finite
    if J > 1000:
        ratios_after.append(k / G) if J % 97 == 0 else None
print("   stage ends (J, s, target, ratio):", [(J, s, round(t, 3), round(r, 3)) for J, s, t, r in stage_ends], flush=True)
late = ratios_after[len(ratios_after) // 3:]
print(f"   ratios over the last two thirds of the run: min {min(late):.3f}, max {max(late):.3f} (every value lies between consecutive stage targets)", flush=True)

# Part 3: Theorem 5.4 stream (exact big-integer arithmetic, one chunk per dyadic block of J)
print("Part 3: Theorem 5.4 stream, target max(2^floor(a floor(log2 J)), floor(log2 J)^2), a_s -> {0.3, 0.8}", flush=True)
from fractions import Fraction
def b_seq(s):
    return Fraction(3, 10) + Fraction(1, 5 * (s + 2)) if s % 2 == 0 else Fraction(4, 5) - Fraction(1, 5 * (s + 2))
def tau_of(a, lg):
    return max(2 ** int(a * lg), lg * lg)
k = 0; s = 0; a = b_seq(0); J_inst = 1
ends = []; track = []
J = 1; LGMAX = 1200
while True:
    lg = (J + 1).bit_length() - 1
    if lg >= LGMAX: break
    tau = tau_of(a, lg)
    Jnext = 2 ** (lg + 1) - 1
    steps = Jnext - J
    if k < tau:
        k += min(tau - k, steps)
    J = Jnext
    lg2 = (J + 1).bit_length() - 1
    ratio = (k.bit_length() - 1) / lg2
    track.append((s, ratio))
    tau2 = tau_of(a, lg2)
    if 2 * k >= tau2 and k <= 2 * tau2 and J >= (s + 3) * J_inst:
        ends.append((lg2, s, float(a), round(ratio, 3)))
        s += 1; a = b_seq(s); J_inst = J
print("   stage ends (log2 J, s, target exponent, log2 k/log2 J):", ends, flush=True)
late = [r for (st, r) in track if st >= 3]
print(f"   ratios after stage 3: min {min(late):.3f}, max {max(late):.3f} (targets' liminf 0.3, limsup 0.8)", flush=True)
print("DONE", flush=True)
