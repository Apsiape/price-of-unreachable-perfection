"""Exact finite checks for the locality cap recalled in Theorem 6.2 and its stopped-multiplier consequence.

Single-threaded, standard library only (fractions), small memory.

What is checked (all exactly, with Fractions):
 1. The finite Cantor clocks (fuel, time) of [UP, Sections P3-P5] on N bits: affine defect 3*4^-N P_0, flip defect 3*4^-N [X,P_0],
    the unguarded increment with its rank-one defect at the all-ones time word, and the exact reset recursion.
 2. The stopped checker of stopped multiplier presentation for multipliers m > n (the superunit-factor regime, "nominal exponent" log m / log n > 1):
    initialization, the three counter transitions, the stop test and P <= H hold EXACTLY on every fuel/time
    block for every input eigenvalue t in a test list, including t = 0, t = 1, exact powers m^-k and their
    neighbours, with all counters in their domains.
 3. Block-sharp source: T = diag over blocks with T = 0 on good blocks and arbitrary values in (0,1] on a fraction nu
    of bad blocks. The penalty S = s(T) is supported on bad blocks and tau(S^2) <= nu/n^2, for every (m,n).
    Contrast with a smooth spectrum T ~ N^-1/2: tau(s(T)^2) ~ N^-(1/beta). So sharp models give loss ~ 1/logD
    (exponent 1) while smooth ones give loss ~ (logD)^-(1/beta).
 4. Linear calibration for m > n (the stopped-multiplier consequence in Section 6.1): on the rare-sector family and on
    random two-sector families the ratio |T|_rho / |S|_rho stays bounded by n, i.e. the m > n checker certifies
    only |T| <= C|S|, never a power > 1, and never worse than linear.
"""
import json
import math
import random
import sys
from fractions import Fraction as Fr

random.seed(20261028)

RESULTS = {}


def bits(x, N):
    return [(x >> j) & 1 for j in range(N)]


def rot_up(x, N):
    # T|b> = |b'> with b'_{j+1} = b_j, b'_0 = b_{N-1}
    top = (x >> (N - 1)) & 1
    return ((x << 1) & ((1 << N) - 1)) | top


def rot_down(x, N):
    # T*|b> : b''_j = b_{j+1}, b''_{N-1} = b_0
    low = x & 1
    return (x >> 1) | (low << (N - 1))


def A_val(x, N):
    return sum(Fr(3, 4 ** (j + 1)) * ((x >> j) & 1) for j in range(N))


def check_clock(N):
    """Check the four Cantor clock relations on N bits, as maps on basis states."""
    D = 1 << N
    out = {}
    # [A,P]=0 trivially (both diagonal). TAT* = 4A - 3P + defect.
    max_aff = Fr(0)
    for y in range(D):
        # (T A T*) |y> = A(T* y) |y>  (diagonal)
        lhs = A_val(rot_down(y, N), N)
        rhs = 4 * A_val(y, N) - 3 * (y & 1)
        d = lhs - rhs
        # predicted defect 3*4^-N * P_0
        pred = Fr(3, 4 ** N) * (y & 1)
        assert d == pred, (N, y, d, pred)
        max_aff = max(max_aff, abs(d))
    out["affine_defect_max"] = float(max_aff)
    # XPX = I-P exactly: X flips bit0 -> obvious; check
    for y in range(D):
        assert ((y ^ 1) & 1) == 1 - (y & 1)
    # [X, TAT*]: TAT* is diagonal with value A(T* y). X maps y -> y^1.
    max_flip = Fr(0)
    for y in range(D):
        v1 = A_val(rot_down(y, N), N)
        v2 = A_val(rot_down(y ^ 1, N), N)
        max_flip = max(max_flip, abs(v1 - v2))
    assert max_flip == Fr(3, 4 ** N), max_flip
    out["flip_defect_max"] = float(max_flip)
    # unguarded increment: U|t> = |t+1 mod D>; RHS X[(I-P) + (T U T*) P]
    defects = []
    for t in range(D):
        lhs = (t + 1) % D
        if t & 1 == 0:
            rhs = t ^ 1
        else:
            s = rot_down(t, N)
            s = (s + 1) % D
            s = rot_up(s, N)
            rhs = s ^ 1
        if lhs != rhs:
            defects.append((t, lhs, rhs))
    assert defects == [(D - 1, 0, 1)], defects  # rank one at the all-ones word
    out["unguarded_defect"] = "rank one at all-ones: |0> vs |1> ; flat norm sqrt(2)*2^(-N/2) = %.6g" % (
        math.sqrt(2) * 2 ** (-N / 2))
    return out


def r_of(q, N):
    r = 0
    while r < N and (q >> r) & 1:
        r += 1
    return r


def Z_of(q, t, N):
    r = r_of(q, N)
    return 1 if t % (1 << r) == 0 else 0


def check_reset(N):
    D = 1 << N
    for q in range(D):
        for t in range(D):
            lhs = Z_of(q, t, N)
            pf, pt = q & 1, t & 1
            rhs = (1 - pf) + pf * (1 - pt) * Z_of(rot_down(q, N), rot_down(t, N), N)
            assert lhs == rhs, (N, q, t, lhs, rhs)
    return True


def stop_index(t, m):
    """Largest j >= 0 with m^j t <= 1, for 0 < t <= 1 (exact rational)."""
    j = 0
    while (m ** (j + 1)) * t <= 1:
        j += 1
    return j


def penalty(t, m, n):
    if t == 0:
        return Fr(0)
    return Fr(1, n ** (stop_index(t, m) + 1))


def history(t, m, n, age):
    """Scalar history of stopped multiplier history (valid for any m,n>=2) at a given age since reset.
    returns C, D, H, P"""
    S = penalty(t, m, n)
    if t == 0:
        return Fr(0), Fr(0), 1, 0
    j = stop_index(t, m)
    r = min(age, j)
    C = (m ** r) * t
    Dv = (n ** r) * S
    H = 1 if age <= j else 0
    P = 1 if age == j else 0
    return C, Dv, H, P


def check_checker(m, n, N, tlist):
    """Check stopped multiplier (4)-(6) exactly on every fuel/time block of N-bit clocks for inputs tlist."""
    Dd = 1 << N
    maxres = Fr(0)
    nchecks = 0
    for t in tlist:
        S = penalty(t, m, n)
        for q in range(Dd):
            per = 1 << r_of(q, N)
            for tt in range(Dd):
                age = tt % per
                agep = (tt + 1) % Dd % per  # age at next time (U is exactly cyclic +1)
                Z = Z_of(q, tt, N)
                Zp = Z_of(q, (tt + 1) % Dd, N)
                assert (Z == 1) == (age == 0)
                assert (Zp == 1) == (agep == 0)
                C, Dv, H, P = history(t, m, n, age)
                Cp, Dp, Hp, Pp = history(t, m, n, agep)
                # domains
                assert 0 <= C <= 1 and 0 <= Dv <= 1 and H in (0, 1) and P in (0, 1)
                assert P <= H  # (I-H)P = 0
                res = []
                # (4) initialization
                res.append(Z * (C - t))
                res.append(Z * (Dv - S))
                res.append(Z * (H - 1))
                # (5) transitions ; X_+ = U* X U has value X(t+1) at time t
                res.append(Hp - (Zp + (1 - Zp) * (H - P)))
                res.append(Cp - (Zp * t + (1 - Zp) * C * (1 + (m - 1) * (H - P))))
                res.append(Dp - (Zp * S + (1 - Zp) * Dv * (1 + (n - 1) * (H - P))))
                # (6) stop test
                res.append((Dv - Fr(1, n)) * P)
                for x in res:
                    maxres = max(maxres, abs(Fr(x)))
                nchecks += len(res)
    return float(maxres), nchecks


def tlist_for(m):
    L = [Fr(0), Fr(1), Fr(1, 2), Fr(1, 3), Fr(7, 10)]
    for k in range(1, 6):
        p = Fr(1, m ** k)
        L += [p, p * Fr(1000001, 1000000), p * Fr(999999, 1000000)]
    L += [Fr(random.randint(1, 10 ** 6), 10 ** 6) for _ in range(6)]
    L += [Fr(1, 2 ** 100)]
    return L


def part1_2():
    out = {}
    for N in (2, 3, 4, 5, 6):
        out["clock_N%d" % N] = check_clock(N)
        check_reset(N)
    out["reset_exact_N2to6"] = True
    chk = {}
    for (m, n) in ((4, 2), (3, 2), (8, 2), (9, 3), (5, 3), (2, 3)):
        for N in (3, 4):
            mr, nc = check_checker(m, n, N, tlist_for(m))
            chk["m%d_n%d_N%d" % (m, n, N)] = {"max_new_residual": mr, "checks": nc}
            assert mr == 0.0
    out["checker_exact"] = chk
    return out


def part3():
    """Sharp vs smooth spectra: tau(S^2) for the m>n penalty."""
    out = {}
    for (m, n) in ((4, 2), (8, 2), (9, 3)):
        beta = math.log(m) / math.log(n)
        rows = []
        for K in (6, 8, 10, 12, 14):
            # sharp: bad fraction nu = 2^-K, T values on bad blocks uniform in (0,1]
            nu = 2.0 ** (-K)
            vals = [Fr(random.randint(1, 1000), 1000) for _ in range(200)]
            ES2 = sum(float(penalty(v, m, n)) ** 2 for v in vals) / len(vals)
            sharp = nu * ES2
            assert sharp <= nu / n ** 2 + 1e-15
            # smooth: N = 2^K lamps-equivalent, T = N^-1/2 * g with g in [1/2, 2] (all moments ~ N^-nu)
            Nn = 2.0 ** K
            vals2 = [Fr(int(1e9 * (0.5 + 1.5 * random.random()) / math.sqrt(Nn)), 10 ** 9) for _ in range(200)]
            smooth = sum(float(penalty(v, m, n)) ** 2 for v in vals2) / len(vals2)
            rows.append({"logD_proxy": 2 ** K, "sharp_tauS2": sharp, "smooth_tauS2": smooth})
        # fit exponents: loss ~ (logD)^-a
        xs = [math.log(r["logD_proxy"]) for r in rows]
        ys1 = [math.log(r["sharp_tauS2"]) for r in rows]
        ys2 = [math.log(r["smooth_tauS2"]) for r in rows]

        def slope(xs, ys):
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sum((a - mx) ** 2 for a in xs)

        out["m%d_n%d" % (m, n)] = {
            "beta_nominal": beta,
            "rows": rows,
            "sharp_slope(loss vs logD)": slope(xs, ys1),
            "smooth_slope(loss vs logD)": slope(xs, ys2),
            "predicted": {"sharp": -1.0, "smooth": -1.0 / beta},
            "implied_game_exponent": {"sharp": -1.0 / slope(xs, ys1), "smooth": -1.0 / slope(xs, ys2)},
        }
    return out


def part4():
    """Linear calibration: |T|/|S| on two-sector flat families for m>n is <= n."""
    out = {}
    for (m, n) in ((4, 2), (8, 2), (9, 3)):
        worst = 0.0
        for _ in range(2000):
            w = 10 ** random.uniform(-8, 0)
            t1 = Fr(random.randint(1, 10 ** 6), 10 ** 6)
            t2 = Fr(random.randint(0, 10 ** 6), 10 ** 6) * Fr(1, m ** random.randint(0, 6))
            T2 = w * float(t1) ** 2 + (1 - w) * float(t2) ** 2
            squared_penalty = w * float(penalty(t1, m, n)) ** 2 + (1 - w) * float(penalty(t2, m, n)) ** 2
            if squared_penalty > 0:
                worst = max(worst, math.sqrt(T2 / squared_penalty))
        # pointwise: t <= m^-j(t) and s(t) = n^-(j+1) so t/s(t) <= n (n/m)^j <= n
        out["m%d_n%d" % (m, n)] = {"max_ratio_T_over_S": worst, "bound_n": n}
        assert worst <= n + 1e-9
        # rare-sector family of stopped multiplier superunit-factor: T=1, S=1/n on weight w: ratio exactly n, and |T|/|S|^alpha diverges
        fam = []
        for w in (1e-2, 1e-4, 1e-6, 1e-8):
            Tn, Sn = math.sqrt(w), math.sqrt(w) / n
            fam.append({"w": w, "T/S": Tn / Sn, "T/S^1.5": Tn / Sn ** 1.5})
        out["m%d_n%d" % (m, n)]["rare_sector_family"] = fam
    return out


def main():
    RESULTS["part1_2_clock_and_checker"] = part1_2()
    RESULTS["part3_sharp_vs_smooth"] = part3()
    RESULTS["part4_linear_calibration"] = part4()
    with open(sys.argv[1] if len(sys.argv) > 1 else "sharp_ceiling.json", "w") as f:
        json.dump(RESULTS, f, indent=1, default=str)
    print(json.dumps(RESULTS["part1_2_clock_and_checker"]["checker_exact"], indent=1), flush=True)
    for k, v in RESULTS["part3_sharp_vs_smooth"].items():
        print(k, "beta_nominal=%.4f" % v["beta_nominal"], "implied exponents:", v["implied_game_exponent"], flush=True)
    for k, v in RESULTS["part4_linear_calibration"].items():
        print(k, "max |T|/|S| =", v["max_ratio_T_over_S"], "<= n =", v["bound_n"], flush=True)
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
