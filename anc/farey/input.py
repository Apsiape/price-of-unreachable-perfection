# Exact-arithmetic check of the Diophantine profile law (Theorem 7.1).
# lam = deep convergent of a continued fraction [2; 1, a2, a3, ...] with a_{n+1} ~ k_n^(r-1), mu = 1 + r.
# G(N) = (a_N - lam)(lam - b_N) at Farey level N; lower ratio log N / log(1/G(N)), upper ratio log N_j / log(1/G(N_{j-1})).
import math
from fractions import Fraction

def lg(x):  # log2 of a positive Fraction or int, exact enough
    if isinstance(x, int):
        return x.bit_length() - 1 + math.log2(x / (1 << (x.bit_length() - 1))) if x > 0 else float('-inf')
    return lg(x.numerator) - lg(x.denominator)

def run(r, nterms, label):
    h = [1, 2]; k = [0, 1]
    for i in range(1, nterms):
        ai = 1 if (i == 1 or k[-1] <= 1) else max(1, 1 << int((r - 1) * lg(k[-1])))
        h.append(ai * h[-1] + h[-2]); k.append(ai * k[-1] + k[-2])
    lam = Fraction(h[-1], k[-1])
    def Gat(n, j):
        p2, q2 = h[n-1] + j*h[n], k[n-1] + j*k[n]
        d1 = Fraction(h[n], k[n]) - lam; d2 = Fraction(p2, q2) - lam
        assert d1 * d2 < 0
        return max(q2, k[n]), -d1 * d2
    lows, ups = [], []
    for n in range(2, len(k) - 3):
        A = (k[n+1] - k[n-1]) // k[n]
        js = {0, 1, A - 1}; t = 1.0
        while t < A:
            js.add(int(t)); t *= 1.3
        for j in sorted(x for x in js if 0 <= x <= A - 1):
            N, G = Gat(n, j)
            lows.append((n, lg(N) / -lg(G)))
            if j >= 1:
                _, Gp = Gat(n, j - 1); ups.append((n, lg(N) / -lg(Gp)))
            elif n >= 3:
                Ap = (k[n] - k[n-2]) // k[n-1]; _, Gp = Gat(n - 1, Ap - 1); ups.append((n, lg(N) / -lg(Gp)))
    mu = 1 + r
    nmax = len(k) - 4
    lo = min(v for n, v in lows if n >= nmax - 1); up = max(v for n, v in ups if n >= nmax - 1)
    print("%-28s mu=%.2f  predicted (lower, upper) = (%.4f, %.4f)   observed at the last two levels = (%.4f, %.4f)"
          % (label, mu, 1 / (mu + 2), (mu - 1) / (2 * mu), lo, up))

run(1.0, 60, "bounded quotients")
run(2.0, 13, "mu=3")
run(3.0, 11, "mu=4")
run(5.0, 9, "mu=6")
run(9.0, 8, "mu=10")
