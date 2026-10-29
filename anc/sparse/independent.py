"""Independent scalar, instruction, noncommuting-column and moment attacks.

Standard library, one thread; this does not simulate compiler/extraction.
No imports from the supplied checker. The supplied JSON instruction tables
are interpreted here with a separately written state machine.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import itertools
import json
import math
import random
import time
from fractions import Fraction as F
from pathlib import Path

OUT = Path(__file__).resolve().parent
RNG = random.Random(20261028)


def first_root_hit(ell):
    lo, hi = 0, (ell + 1) ** 2
    while lo < hi:
        mid = (lo + hi) // 2
        if mid ** 3 >= ell ** 4:
            hi = mid
        else:
            lo = mid + 1
    return lo


def interpret(filename, wanted, root):
    obj = json.loads((OUT / filename).read_text())
    code = obj['instructions']
    regs = [0] * len(obj['registers'])
    pc = age = pulses = ones = 0
    stage, length, pos = 1, 2, 0
    # Keep a bounded sample including every power-of-two reset boundary,
    # a dense short prefix, and neighborhoods of every emitted instruction.
    selected = set(range(512))
    for power in range(1, 26):
        selected.update((2 ** power - 1, 2 ** power))
    selected.update(RNG.randrange(1, 3000000) for _ in range(1200))
    rows, pulse_rows, hits, pulse_times = {}, [], {0: 0}, []
    while pulses < wanted:
        ins = code[pc]
        output = ins[0] == 'NOP' and ins[2] is not None
        if root and (age in selected or output):
            rows[age] = (pc, tuple(regs), pulses, ones)
        if output:
            if root:
                pulse_rows.append(age)
                selected.add(age + 1)
            bit = ins[2]
            pulses += 1
            ones += bit
            if root:
                assert ones ** 4 <= pulses ** 3 < (ones + 1) ** 4
                assert regs[0] == pulses ** 3
                assert regs[3] == (ones + 1) ** 4
            else:
                assert bit == (pos % stage == 0)
                pos += 1
                if pos == length:
                    stage += 1
                    length *= 2
                    pos = 0
            pulse_times.append(age + 1)
            if bit:
                hits[ones] = age + 1
        if ins[0] == 'INC':
            regs[ins[1]] += 1
            pc = ins[2]
        elif ins[0] == 'DECJZ':
            reg = ins[1]
            if regs[reg] == 0:
                pc = ins[2]
            else:
                regs[reg] -= 1
                pc = ins[3]
        else:
            pc = ins[1]
        age += 1
        if root and (age - 1 in selected or output):
            rows[age] = (pc, tuple(regs), pulses, ones)
    if root:
        rows[age] = (pc, tuple(regs), pulses, ones)
    return code, rows, hits, pulse_times, {'outputs': pulses, 'ones': ones,
        'elementary_instructions': age, 'controls': len(code),
        'registers': len(regs)}


def root_histories(code, rows, hits, pulse_times):
    # Only existing ages from the compact sample are required.
    total = 0
    tests = [(F(0), None), (F(1), 0), (F(3, 4), 0)]
    for ell in (1, 2, 3, 4, 7, 10, 15, 20, 50):
        tests.extend([(F(1, 2 ** ell), ell),
                      (F(3, 2 ** (ell + 2)), ell)])
    max_age = max(rows)
    periods = [2 ** p for p in range(1, 25) if 2 ** p <= max_age]

    def value(h, t, ell):
        _, _, j, k = rows[h]
        if ell is None:
            return F(0), F(0), 1, 0, F(0)
        p = first_root_hit(ell)
        stop_age = hits.get(ell, max_age + 1)
        return (t * 2 ** min(k, ell), F(2 ** min(j, p), 2 ** (p + 1)),
                int(h <= stop_age), int(h == stop_age), F(1, 2 ** (p + 1)))

    candidates = [(h, h + 1, 0) for h in rows if h + 1 in rows]
    candidates += [(period - 1, 0, 1) for period in periods
                   if period - 1 in rows]
    for h, hp, reset in candidates:
        pc, regs, _, _ = rows[h]
        pcp, regsp, _, _ = rows[hp]
        ins = code[pc]
        a, ap = F(1, h + 1), F(1, hp + 1)
        assert ap * (1 + a) == a + reset
        next_pc = ins[2] if ins[0] == 'INC' else (
            ins[2] if ins[0] == 'DECJZ' and regs[ins[1]] == 0 else
            ins[3] if ins[0] == 'DECJZ' else ins[1])
        assert pcp == (0 if reset else next_pc)
        for i, ni in enumerate(regs):
            inc = int(ins[0] == 'INC' and ins[1] == i)
            dec = int(ins[0] == 'DECJZ' and ins[1] == i and ni > 0)
            assert F(regsp[i], hp + 1) * (1 + a) == (1-reset) * (ni*a + (inc-dec)*a)
            assert ni <= h
            e = int(ni == 0)
            y, w = ni * a, max(ni - 1, 0) * a
            assert y * e == 0 and (y-a)*(1-e) == w*(1-e)
        is_pulse = ins[0] == 'NOP' and ins[2] is not None
        is_one = is_pulse and ins[2] == 1
        for t, ell in tests:
            c, d, running, stop, s = value(h, t, ell)
            cp, dp, rp, _, _ = value(hp, t, ell)
            assert 0 <= c <= 1 and 0 <= d <= F(1, 2)
            assert (1-running)*stop == 0 and (d-F(1, 2))*stop == 0
            assert rp == reset + (1-reset)*(running-stop)
            assert cp == reset*t + (1-reset)*c*(1+is_one*(running-stop))
            assert dp == reset*s + (1-reset)*d*(1+is_pulse*(running-stop))
            total += 1
    return {'exact_sampled_transitions': total, 'sampled_ages': len(rows),
            'reset_periods': periods, 'input_values': len(tests),
            'verified_stop_levels': sorted(hits)}


def eye(d):
    return [[F(i == j) for j in range(d)] for i in range(d)]


def add(a, b, sign=1):
    return [[x + sign*y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, c):
    return [[c*x for x in row] for row in a]


def mul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def transpose(a):
    return list(map(list, zip(*a)))


def projection(d):
    v = [RNG.randrange(-3, 4) for _ in range(d)]
    if not any(v):
        v[0] = 1
    den = sum(x*x for x in v)
    return [[F(x*y, den) for y in v] for x in v]


def column_and_identity_checks():
    d, I = 3, eye(3)
    noncommuting, identities, columns = 0, 0, 0
    for trial in range(8):
        K = I
        gram = scale(I, 0)
        for h in range(7):
            q, e, p = (projection(d) for _ in range(3))
            noncommuting += int(mul(q, e) != mul(e, q))
            badc = mul(add(I, q, -1), K)
            V = mul(q, K)
            badz = mul(add(I, e, -1), V)
            good = mul(e, V)
            stop = mul(p, good)
            K = mul(add(I, p, -1), good)
            for col in (badc, badz, stop):
                gram = add(gram, mul(transpose(col), col))
                columns += 1
        gram = add(gram, mul(transpose(K), K))
        assert gram == I
    for _ in range(30):
        A, Ap, Y, Yp = [scale(projection(d), F(RNG.randrange(1, 10), 10)) for _ in range(4)]
        a, b = F(2, 7), F(3, 8)
        for s in (-1, 0, 1):
            ap, bp = a/(1+a), (b+s*a)/(1+a)
            lhs = scale(add(Ap, scale(I, ap), -1), 1+a)
            rhs = add(add(mul(Ap, add(I, A)), A, -1), mul(add(I, Ap, -1), add(A, scale(I, a), -1)))
            assert lhs == rhs
            lhs = scale(add(Yp, scale(I, bp), -1), 1+a)
            bracket = add(add(mul(Yp, add(I, A)), Y, -1), scale(A, s), -1)
            rhs = add(add(bracket, add(Y, scale(I, b), -1)), mul(add(scale(I, s), Yp, -1), add(A, scale(I, a), -1)))
            assert lhs == rhs
            identities += 2
    return {'column_decompositions': 8, 'discarded_columns': columns,
            'noncommuting_pairs': noncommuting, 'ordered_identities': identities,
            'arithmetic': 'exact rational matrices'}


def scalar_envelopes():
    words = prefixes = 0
    for length in range(1, 14):
        for bits in itertools.product((0, 1), repeat=length):
            j = k = prev = 0
            for bit in bits:
                j += 1
                k += bit
                assert j-k >= prev
                prev = j-k
                assert 2 ** (j+1-k) <= 2 ** (length+1-sum(bits))
                prefixes += 1
            words += 1
    # Piecewise-linear objective: its maximum over integer ell is within
    # a small neighborhood of the real optimizer, plus the endpoint.
    for r in range(1, 201):
        center = F(27*r**3, 64)
        candidates = {0} | set(range(max(0, int(center)-8), int(center)+10))
        for ell in candidates:
            p = first_root_hit(ell)
            assert p**3 >= ell**4 and (p == 0 or (p-1)**3 < ell**4)
            assert r*(ell+1)-p-1 <= F(27*r**4, 256)+r-1
    # Quantified cutoff arithmetic; no sampling of unknown imported constants.
    for n in (100, 1000, 10000, 100000, 1000000):
        for degree in (1, 3, 7):
            K = degree + 3
            j = max(0, n - K*(n+1).bit_length())
            if j:
                assert 2 ** j * (j+1) ** degree <= 2 ** n
    return {'all_binary_words': words, 'prefix_inequalities': prefixes,
            'root_envelope_orders': 200, 'includes_zero_words': True}


def field_mul(x, y):
    ans = 0
    while y:
        if y & 1:
            ans ^= x
        y >>= 1
        x <<= 1
        if x & 8:
            x ^= 11  # z^3+z+1, GF(8)
    return ans


def character_checks():
    rows = []
    for r in (1, 2, 3):
        n0, N = 1+3*(2*r-1), 8
        labels = []
        for alpha in range(N):
            label, power = 1, 1
            for i in range(1, 2*r):
                power = field_mul(power, alpha)
                label |= power << (1+3*(i-1))
            labels.append(label)
        for mask in range(1, 1 << N):
            if mask.bit_count() <= 2*r:
                xor = 0
                for i, label in enumerate(labels):
                    if mask >> i & 1:
                        xor ^= label
                assert xor != 0
        total = 0
        for u in range(1 << n0):
            sigsum = sum(1 if (label & u).bit_count() % 2 == 0 else -1 for label in labels)
            total += sigsum ** (2*r)
        pairings = math.prod(range(1, 2*r, 2))
        assert total <= (1 << n0)*pairings*N**r
        # Removing the all-positive u=0 cannot increase the average.
        assert F(total-N**(2*r), (1 << n0)-1) <= F(total, 1 << n0)
        rows.append({'r': r, 'N': N, 'character_space_dimension': n0,
                     'enumerated_characters': 1 << n0,
                     'exact_sum_power_moment': total})
    # Conservative global uniform dimension envelope from the explicit formula.
    dim_cases = 0
    for exponent in range(1, 301):
        N = 2 ** exponent
        for r in (1, 2, 3, 8, 32, 128, 512):
            n0 = 1+(2*r-1)*exponent
            fourth = math.isqrt(math.isqrt(N))
            if fourth**4 < N:
                fourth += 1
            M = max(48, fourth, (n0-4+3)//4)
            n = 4*M+4
            upper_log_dimension = 3*N+n*n+2*M.bit_length()
            assert upper_log_dimension <= 65536*(r+1)**2*N
            dim_cases += 1
    return {'finite_field_moments': rows, 'uniform_dimension_cases': dim_cases,
            'dimension_envelope_B': 65536, 'dimension_envelope_q': 2}


def finite_clock_checks():
    cases = reset_cases = 0
    for bits in range(1, 7):
        dim, mask = 2**bits, 2**bits-1
        def rotate(x):
            return (x >> 1) | ((x & 1) << (bits-1))
        def unrotate(x):
            return ((x << 1) & mask) | (x >> (bits-1))
        def age_period(fuel):
            count = 0
            while count < bits and fuel >> count & 1:
                count += 1
            return 2**count
        def reset(fuel, t):
            return int(t % age_period(fuel) == 0)
        def cantor(x):
            return sum((F(3, 4**(i+1)) for i in range(bits) if x >> i & 1), F(0))
        bad_columns = []
        for t in range(dim):
            tail = cantor(rotate(t))
            affine_defect = tail-4*cantor(t)+3*(t & 1)
            assert affine_defect == F(3*(t & 1), 4**bits)
            if t & 1:
                branch = unrotate((rotate(t)+1) % dim)
            else:
                branch = t
            increment_prediction = branch ^ 1
            if increment_prediction != (t+1) % dim:
                bad_columns.append(t)
            cases += 1
        assert bad_columns == [dim-1]
        # Each of the two distinct entries in the bad column has modulus one.
        assert F(2, dim) == 2*F(1, 2**bits)
        for f in range(dim):
            for t in range(dim):
                lhs = reset(f, t)
                rhs = 1-(f & 1)+(f & 1)*(1-(t & 1))*reset(rotate(f), rotate(t))
                assert lhs == rhs
                period = age_period(f)
                assert ((t+1) % dim) % period == ((t % period)+1) % period
                reset_cases += 1
    # Exact stage counts at enormously larger prefixes, without simulation.
    stage_samples, completed = 0, 0
    largest_scaled_error = 0.0
    for m in range(1, 151):
        L = 2**m
        for prefix in (0, 1, L//4, L//2, 3*L//4, L):
            j = L-2+prefix
            k = completed+(prefix+m-1)//m
            if m >= 8:
                scaled_error = abs(k*m-j)*m/L
                largest_scaled_error = max(largest_scaled_error, scaled_error)
                assert abs(k*m-j)*m <= 20*L
            stage_samples += 1
        completed += (L+m-1)//m
    return {'clock_column_cases': cases, 'reset_basis_cases': reset_cases,
            'clock_bits': [1, 2, 3, 4, 5, 6],
            'large_stage_prefixes': stage_samples,
            'max_scaled_stage_error_from_stage_8': largest_scaled_error}



def main():
    start = time.monotonic()
    code, rows, hits, pulse_times, root = interpret('sparse_program.json', 48, True)
    history = root_histories(code, rows, hits, pulse_times)
    del rows
    _, _, _, _, near = interpret('near_exponential_program.json', 200000, False)
    result = {'status': 'PASS', 'root_machine': root, 'root_histories': history,
              'near_machine': near, 'matrix_checks': column_and_identity_checks(),
              'scalar_checks': scalar_envelopes(), 'source_moment_checks': character_checks(),
              'finite_clocks_and_stages': finite_clock_checks(),
              'elapsed_seconds': time.monotonic()-start,
              'scope': 'finite checks; no reconstruction of compiler or distillation theorem'}
    (OUT/'independent-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
