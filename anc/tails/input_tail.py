"""Finite diagnostics for the uniform-tail source (Lemma 3.1).

For a list zeta_1..zeta_N in F_2^n, c(u) = N^-1 sum_j (-1)^<zeta_j,u>, computed for ALL u by a fast
Walsh-Hadamard transform of the list's histogram. Compares
  (a) an i.i.d. uniform list: avg_{u!=0} exp(N c(u)^2/4) <= sqrt 2 in expectation (Lemma 3.1), the tail
      fraction{|c|>kappa} <= sqrt2 exp(-N kappa^2/4), and moments avg|c|^{2nu} <= sqrt2 (4nu/(eN))^nu
      uniformly in nu;
  (b) the 2k-wise independent Vandermonde-type list (first coordinate 1), which has c(u*) = -1 at
      u* = e_first, so its high moments are >= 2^-n / (2^n - 1)... i.e. NOT uniform in nu.
Single-threaded; arrays of size 2^n with n <= 18.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, math

def fwht(a):
    a = a.astype(np.float64).copy(); h = 1; n = a.shape[0]
    while h < n:
        a = a.reshape(-1, 2, h)
        x = a[:, 0, :].copy(); y = a[:, 1, :].copy()
        a[:, 0, :] = x + y; a[:, 1, :] = x - y
        a = a.reshape(n); h *= 2
    return a

def cvals(zetas, n):
    hist = np.bincount(zetas, minlength=1 << n)
    return fwht(hist)[1:] / len(zetas)   # drop u = 0

rng = np.random.default_rng(20261028)
n = 16
print(f"n = {n}, number of nonzero u = {(1<<n)-1}", flush=True)
allok = True
for N in (64, 256, 1024, 4096):
    stats = []
    for trial in range(5):
        z = rng.integers(0, 1 << n, size=N)
        c = cvals(z, n)
        mgf = float(np.mean(np.exp(N * c * c / 4)))
        stats.append(mgf)
    z = rng.integers(0, 1 << n, size=N); c = cvals(z, n)
    tails = []
    for kap in (2 / math.sqrt(N), 3 / math.sqrt(N), 4 / math.sqrt(N), 5 / math.sqrt(N)):
        frac = float(np.mean(np.abs(c) > kap)); bnd = math.sqrt(2) * math.exp(-N * kap * kap / 4)
        tails.append((round(kap * math.sqrt(N), 1), frac, bnd))
    mom = []
    for nu in (1, 2, 4, 8, 16):
        m = float(np.mean(np.abs(c) ** (2 * nu))); b = math.sqrt(2) * (4 * nu / (math.e * N)) ** nu
        mom.append((nu, m / b))
    print(f"N={N}: E-list avg exp(Nc^2/4) over 5 lists = {[round(s,3) for s in stats]} (bound sqrt2=1.414 in mean)", flush=True)
    print(f"   tail (kappa*sqrtN, fraction, bound) = {[(k, f'{f:.2e}', f'{b:.2e}') for k,f,b in tails]}", flush=True)
    print(f"   moment/bound ratios (nu, ratio) = {[(nu, round(r,3)) for nu,r in mom]}", flush=True)
    allok &= all(f <= b for _, f, b in tails[1:]) and all(r <= 1.0 for _, r in mom)

# (b) Vandermonde-type 2k-wise independent list with first coordinate 1 (bounded-independence construction, k=1 example):
# zeta_j = (1, alpha_j) in F_2 + F_{2^m}; here m = n-1 bits, distinct alpha_j.
N = 1024
alphas = rng.choice(1 << (n - 1), size=N, replace=False)
z = (alphas << 1) | 1          # lowest bit = the 'first coordinate' 1
c = cvals(z, n)
ustar = 1                      # u* = indicator of the first coordinate
print(f"bounded-independence list (first coord 1): c(u*) = {c[ustar-1]:.3f} (norm-one exceptional block, mass 1/(2^n-1))", flush=True)
for nu in (1, 2, 4, 8, 16, 32):
    m = float(np.mean(np.abs(c) ** (2 * nu)))
    print(f"   nu={nu}: avg|c|^(2nu) * N^nu = {m * N**nu:.3e}   (uniform list would give <= sqrt2 (4nu/e)^nu = {math.sqrt(2)*(4*nu/math.e)**nu:.3e})", flush=True)
print("PASS" if allok else "FAIL", flush=True)
