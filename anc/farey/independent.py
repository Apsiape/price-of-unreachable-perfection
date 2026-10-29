"""Independent finite checks for Farey arithmetic. Standard library, one thread, no input imports.

Exact arithmetic verifies finite identities, not asymptotic or presentation proofs.
All writes are confined to this file's output directory.
"""
from fractions import Fraction as F
from itertools import product
from collections import deque
import math
import random
import time


def cf_value(digits):
    x = F(digits[-1])
    for a in reversed(digits[:-1]):
        x = a + 1 / x
    return x


def brute_neighbors(x, N):
    lower, upper = None, None
    for q in range(1, N + 1):
        p = (x.numerator * q) // x.denominator
        lo, hi = F(p, q), F(p + 1, q)
        assert lo < x < hi
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else min(upper, hi)
    return lower, upper


def mediant_neighbors(x, N):
    m = x.numerator // x.denominator
    p, q, r, s = m, 1, m + 1, 1
    while q + s <= N:
        t = F(p + r, q + s)
        assert t != x
        if t < x:
            p, q = p + r, q + s
        else:
            r, s = p + r, q + s
    return F(p, q), F(r, s)


def farey_checks():
    rng = random.Random(717)
    digits_list = [
        [2] + [1] * 22,
        [2, 1, 47, 1, 1, 103, 2, 1, 1, 1, 257, 1, 11],
        [2, 9, 1, 1, 1, 1, 61, 1, 3, 1, 1, 409, 2],
    ]
    digits_list += [[2, 1] + [rng.randrange(1, 25) for _ in range(10)] for _ in range(17)]
    targets = [cf_value(a) for a in digits_list]
    # Include points close to both ends of the KRS interval.
    targets += [F(138203, 100003), F(361813, 100003)]
    assert all(x * x - 5 * x + 5 < 0 for x in targets)
    exterior_chain, v, dim = set(), F(1), F(1)
    while v.denominator <= 80:
        exterior_chain.add(v)
        dim *= 4 - v
        v = 1 + 1 / (4 - v)
        assert dim == v.denominator
    checked = 0
    for x in targets:
        old_g = None
        for N in range(1, 81):
            b, a = brute_neighbors(x, N)
            assert (b, a) == mediant_neighbors(x, N)
            assert a.numerator * b.denominator - b.numerator * a.denominator == 1
            assert a.denominator + b.denominator > N
            assert a.denominator * b.denominator >= N
            if b * b - 5 * b + 5 > 0:
                assert b in exterior_chain
            if a * a - 5 * a + 5 > 0:
                assert 5 - a in exterior_chain
            g = (a - x) * (x - b)
            assert g > 0 and g <= F(1, 4 * N * N)
            assert old_g is None or g <= old_g
            old_g = g
            checked += 1
    print(f'PASS Farey: {checked} exhaustive denominator levels; mediant agreement, determinants, monotonicity, universal bound')
    print('PASS coarse-window coverage: every exterior neighbor is a Coxeter-seed value or its complement; seed dimensions equal reduced denominators')

    formula_count = 0
    for digits in digits_list:
        x = cf_value(digits)
        pm, p, qm, q = 1, digits[0], 0, 1
        for n in range(1, len(digits) - 2):
            pm, p = p, digits[n] * p + pm
            qm, q = q, digits[n] * q + qm
            t = cf_value(digits[n + 1:])
            r = F(qm, q)
            for j in range(digits[n + 1]):
                d = qm + j * q
                N = max(q, d)
                if N > 600:
                    continue
                b, a = sorted([F(p, q), F(pm + j * p, d)])
                assert (b, a) == brute_neighbors(x, N)
                predicted = (t - j) / (q ** 4 * (j + r) * (t + r) ** 2)
                assert predicted == (a - x) * (x - b)
                # The same pair survives until its next mediant is available.
                Nend = qm + (j + 1) * q - 1
                assert (b, a) == mediant_neighbors(x, Nend)
                formula_count += 1
    print(f'PASS continued-fraction product formula: {formula_count} exact semiconvergent steps, including irregular large quotients')

    convex_count = 0
    for N in range(1, 10):
        vals = sorted({F(p, q) for q in range(1, N + 1) for p in range(5 * q + 1)})
        for x in [F(223607, 100003), F(249999, 100003), F(140009, 100003)]:
            b, a = brute_neighbors(x, N)
            g = (a - x) * (x - b)
            for u in vals:
                if u >= x:
                    break
                for v in vals:
                    if v <= x:
                        continue
                    w = (v - x) / (v - u)
                    var = w * (u - x) ** 2 + (1 - w) * (v - x) ** 2
                    assert var == (v - x) * (x - u) >= g
                    convex_count += 1
    print(f'PASS variance lower bound: {convex_count} exact two-support mixtures; near neighbors minimize the variance')


def krs_checks():
    count, transport_count = 0, 0
    for q in range(2, 81):
        for p in range((3 * q) // 2 + 1, 2 * q):
            if math.gcd(p, q) != 1:
                continue
            alpha, eps = F(p, q), F(p - q, q)
            b, sumk = alpha, 0
            for _ in range(p - q):
                legal = [k for k in (1, 2, 3) if 0 < 3 - b - k * eps <= eps]
                assert len(legal) == 1 and 1 <= b <= alpha
                k = legal[0]
                residue = 3 - b - k * eps
                sumk += k
                b = alpha - residue
            assert residue == eps
            assert sumk == 3 * q - p - 1
            assert sumk + p - q + 1 == 2 * q
            count += 1
    for q0 in range(1, 81):
        for p0 in range(q0, 4 * q0 + 1):
            if math.gcd(p0, q0) != 1 or p0 * p0 - 5 * p0 * q0 + 5 * q0 * q0 >= 0:
                continue
            x, dim, steps = F(p0, q0), F(2 * q0), 0
            while not F(3, 2) <= x <= 2:
                if x > 3:
                    x = 5 - x
                elif x > 2:
                    dim *= 4 - x
                    x = 1 + 1 / (4 - x)
                else:
                    oldq = x.denominator
                    dim *= x - 1
                    x = 4 - 1 / (x - 1)
                    assert x.denominator < F(oldq, 2)
                assert dim == 2 * x.denominator
                steps += 1
                assert steps < 30
            assert dim == 2 * x.denominator
            transport_count += 1
    print(f'PASS KRS dimension bookkeeping: {count} reduced base values; {transport_count} whole-window transports')


def central_product_checks():
    def mul(g, h):
        c, a, b = g
        C, A, B = h
        return c ^ C ^ ((b & A).bit_count() & 1), a ^ A, b ^ B
    triples = 0
    for sites in [1, 2, 3]:
        elems = list(product(range(2), range(1 << sites), range(1 << sites)))
        assert len(elems) == 2 * 4 ** sites == 8 ** sites // (2 ** (sites - 1))
        identity, J = (0, 0, 0), (1, 0, 0)
        central = [g for g in elems if all(mul(g, h) == mul(h, g) for h in elems)]
        assert central == [identity, J]
        for i in range(sites):
            x, z = (0, 1 << i, 0), (0, 0, 1 << i)
            assert mul(x, x) == mul(z, z) == identity
            assert mul(mul(x, z), mul(x, z)) == J
            for j in range(sites):
                if i != j:
                    w = (0, 0, 1 << j)
                    assert mul(x, w) == mul(w, x)
        if sites <= 2:
            for g, h, k in product(elems, repeat=3):
                assert mul(mul(g, h), k) == mul(g, mul(h, k))
                triples += 1
    print(f'PASS Pauli central products: 1,2,3 sites; orders, normal forms, centers, lamp relations; {triples} associative triples')


def affine_checks():
    dimension = 3
    points = range(1 << dimension)
    identity = tuple(points)
    def compose(g, h):
        return tuple(g[h[v]] for v in points)
    xs = {(i, j): tuple(v ^ (((v >> j) & 1) << i) for v in points)
          for i in range(dimension) for j in range(dimension) if i != j}
    ts = {i: tuple(v ^ (1 << i) for v in points) for i in range(dimension)}
    gens = list(xs.values()) + list(ts.values())
    seen, queue = {identity}, deque([identity])
    while queue:
        g = queue.popleft()
        for h in gens:
            gh = compose(g, h)
            if gh not in seen:
                seen.add(gh)
                queue.append(gh)
    assert len(seen) == 1344
    assert sum(g[0] == 0 for g in seen) == 168
    assert len({(g[0], g[1]) for g in seen}) == 8 * 7
    for (i, j), x in xs.items():
        assert compose(x, x) == identity
        swap = compose(compose(x, xs[j, i]), x)
        assert swap == tuple(v ^ ((((v >> i) ^ (v >> j)) & 1) * ((1 << i) | (1 << j))) for v in points)
        for k in range(dimension):
            expected = compose(ts[i], ts[j]) if k == j else ts[k]
            assert compose(compose(x, ts[k]), x) == expected
        for (k, l), y in xs.items():
            if i != l and j != k:
                assert compose(x, y) == compose(y, x)
        for k in range(dimension):
            if k != i and k != j:
                y = xs[j, k]
                assert compose(compose(compose(x, y), x), y) == xs[i, k]
    print('PASS affine presentation identities: AGL(3,2) has 1344 elements, 168-element zero stabilizer, double transitivity, all displayed relations')

    hashes = []
    for columns in product(range(4), repeat=dimension):
        for offset in range(4):
            hashes.append(tuple(offset ^ (columns[0] if v & 1 else 0)
                                ^ (columns[1] if v & 2 else 0)
                                ^ (columns[2] if v & 4 else 0) for v in points))
    assert len(set(hashes)) == 256
    for v in points:
        for w in points:
            if v != w:
                assert sum(eta[v] == eta[w] for eta in hashes) == 64
    hashset = set(hashes)
    for g in gens:
        assert {tuple(eta[g[v]] for v in points) for eta in hashes} == hashset
    # Check the multiplication defect's central sign for all masks supported on two sites.
    def push(mask, eta):
        out = 0
        for v in points:
            if mask >> v & 1:
                out ^= 1 << eta[v]
        return out
    support = [0, 1, 2, 3]
    checks = 0
    for eta in hashes:
        if eta[0] != eta[1]:
            for b in support:
                for a in support:
                    assert ((push(b, eta) & push(a, eta)).bit_count() & 1) == ((a & b).bit_count() & 1)
                    checks += 1
    constant = hashes[0]
    assert ((push(1, constant) & push(2, constant)).bit_count() & 1) != ((1 & 2).bit_count() & 1)
    print(f'PASS affine hashes: 256 hashes, all 56 ordered collision probabilities exactly 1/4, covariance, {checks} Pauli sign checks; collision negative control detected')


if __name__ == '__main__':
    started = time.perf_counter()
    farey_checks()
    krs_checks()
    central_product_checks()
    affine_checks()
    print(f'ALL INDEPENDENT CHECKS PASSED ({time.perf_counter() - started:.2f} seconds, one thread)')
