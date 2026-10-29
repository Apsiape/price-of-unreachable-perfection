"""Finite checks for sparse-profile. Standard library only; one process and one thread.

This is not a simulation of the imported lamp source, compiler or distillation.
It checks the new sparse counter program, its scalar histories, and envelopes.
"""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
import json
import math
import time
from fractions import Fraction as Q
from pathlib import Path

OUT = Path(__file__).resolve().parent


class Builder:
    def __init__(self):
        self.code = []
        self.labels = {}

    def mark(self, name):
        assert name not in self.labels
        self.labels[name] = len(self.code)

    def inc(self, reg, nxt=None):
        self.code.append(("INC", reg, nxt))

    def dec(self, reg, zero, nonzero=None):
        self.code.append(("DECJZ", reg, zero, nonzero))

    def nop(self, nxt=None, bit=None):
        self.code.append(("NOP", nxt, bit))

    def add(self, src, dst, tmp, tag):
        """dst += src, preserving src and clearing an initially zero tmp."""
        self.mark(tag + "_scan")
        self.dec(src, tag + "_restore")
        self.inc(dst)
        self.inc(tmp, tag + "_scan")
        self.mark(tag + "_restore")
        self.dec(tmp, tag + "_done")
        self.inc(src, tag + "_restore")
        self.mark(tag + "_done")

    def clear(self, reg, tag):
        self.mark(tag)
        self.dec(reg, tag + "_done", tag)
        self.mark(tag + "_done")

    def resolve(self):
        def target(value, i):
            return i + 1 if value is None else self.labels[value]
        code = []
        for i, ins in enumerate(self.code):
            if ins[0] == "INC":
                code.append((ins[0], ins[1], target(ins[2], i)))
            elif ins[0] == "DECJZ":
                code.append((ins[0], ins[1], target(ins[2], i), target(ins[3], i)))
            else:
                code.append((ins[0], target(ins[1], i), ins[2]))
        assert all(0 <= x < len(code) for ins in code for x in
                   ([ins[2]] if ins[0] == "INC" else
                    [ins[2], ins[3]] if ins[0] == "DECJZ" else [ins[1]]))
        return code


REGS = ["cube", "cube_d1", "cube_d2", "threshold", "fourth_d1",
        "fourth_d2", "fourth_d3", "tmp", "left", "right"]
RI = {name: i for i, name in enumerate(REGS)}


def make_program():
    b = Builder()
    for name, value in [("cube_d1", 1), ("cube_d2", 6), ("threshold", 1),
                        ("fourth_d1", 15), ("fourth_d2", 50),
                        ("fourth_d3", 60)]:
        for _ in range(value):
            b.inc(RI[name])
    b.mark("next_j")
    b.add(RI["cube_d1"], RI["cube"], RI["tmp"], "cube_update")
    b.add(RI["cube_d2"], RI["cube_d1"], RI["tmp"], "d1_update")
    for _ in range(6):
        b.inc(RI["cube_d2"])
    b.clear(RI["left"], "clear_left")
    b.clear(RI["right"], "clear_right")
    b.add(RI["cube"], RI["left"], RI["tmp"], "copy_left")
    b.add(RI["threshold"], RI["right"], RI["tmp"], "copy_right")
    b.mark("compare")
    b.dec(RI["right"], "one")
    b.dec(RI["left"], "zero", "compare")
    b.mark("one")
    b.add(RI["fourth_d1"], RI["threshold"], RI["tmp"], "fourth_update")
    b.add(RI["fourth_d2"], RI["fourth_d1"], RI["tmp"], "fourth_d1_update")
    b.add(RI["fourth_d3"], RI["fourth_d2"], RI["tmp"], "fourth_d2_update")
    for _ in range(24):
        b.inc(RI["fourth_d3"])
    b.nop("next_j", bit=1)
    b.mark("zero")
    b.nop("next_j", bit=0)
    return b.resolve(), b.labels


def advance(code, pc, counters):
    ins = code[pc]
    if ins[0] == "INC":
        counters[ins[1]] += 1
        return ins[2]
    if ins[0] == "DECJZ":
        if counters[ins[1]] == 0:
            return ins[2]
        counters[ins[1]] -= 1
        return ins[3]
    return ins[1]


def root4(value):
    return math.isqrt(math.isqrt(value))


def counter_check(code, pulses=40, retain=8193):
    pc, counters, steps, total = 0, [0] * len(REGS), 0, 0
    trace, pulse_times, bits = [], [], []
    max_counter = 0
    while len(bits) < pulses:
        if steps < retain:
            trace.append((pc, tuple(counters), len(bits), total))
        ins = code[pc]
        if ins[0] == "NOP" and ins[2] is not None:
            bits.append(ins[2])
            total += ins[2]
            j = len(bits)
            assert total == root4(j ** 3), (j, total)
            assert counters[RI["cube"]] == j ** 3
            assert counters[RI["threshold"]] == (total + 1) ** 4
            pulse_times.append(steps + 1)
        pc = advance(code, pc, counters)
        steps += 1
        assert min(counters) >= 0
        if steps < retain or steps % 1009 == 0:
            assert max(counters) <= steps
        max_counter = max(max_counter, max(counters))
    table = [{"j": j, "k": root4(j ** 3), "instructions": pulse_times[j - 1],
              "instructions_over_j_plus_1_pow_4": pulse_times[j - 1] / (j + 1) ** 4}
             for j in (1, 2, 4, 8, 16, 32, pulses)]
    return trace, {"controls": len(code), "registers": len(REGS),
                   "pulses": pulses, "instructions": steps,
                   "largest_counter": max_counter,
                   "max_observed_time_ratio": max(t / (j + 2) ** 4
                                                  for j, t in enumerate(pulse_times)),
                   "table": table}


def exact_history_checks(code, trace):
    """controller (11),(12),(22), including every period-end reset, exactly."""
    checks = 0
    inputs = [Q(0), Q(1), Q(1, 2), Q(3, 4), Q(1, 4), Q(1, 8),
              Q(1, 2) - Q(1, 2 ** 16), Q(1, 2) + Q(1, 2 ** 16), Q(1, 2 ** 50)]

    def history(age, t):
        _, _, j, k = trace[age]
        stop = None
        if t:
            ell = 0
            while t * 2 ** (ell + 1) <= 1:
                ell += 1
            if ell == 0:
                stop = 0
            else:
                for h, row in enumerate(trace):
                    if row[3] == ell:
                        stop = h
                        break
        # If no stop has been seen, use a legal remote stop to define S.
        # For dyadic t with ell beyond trace, its true pulse index is explicit.
        if t:
            ell = 0
            while t * 2 ** (ell + 1) <= 1:
                ell += 1
            pulse_index = 0 if ell == 0 else next_pulse_index(ell)
            s = Q(1, 2 ** (pulse_index + 1))
        else:
            s = Q(0)
        before = stop is None or age <= stop
        stopped_age = age if before else stop
        _, _, js, ks = trace[stopped_age]
        return (t * 2 ** ks, s * 2 ** js, int(before), int(stop == age), s)

    for period in (2, 8, 32, 128, 512, 2048):
        assert len(trace) > period
        for h in range(period):
            hp = (h + 1) % period
            reset = int(hp == 0)
            pc, ns, _, _ = trace[h]
            pc_next, ns_next, _, _ = trace[hp]
            a, ap = Q(1, h + 1), Q(1, hp + 1)
            assert ap * (1 + a) == a + reset
            out = [0] * len(ns)
            if code[pc][0] == "INC":
                out[code[pc][1]] = 1
            elif code[pc][0] == "DECJZ" and ns[code[pc][1]] > 0:
                out[code[pc][1]] = -1
            for i, value in enumerate(ns):
                y, yp = value * a, ns_next[i] * ap
                e, w = int(value == 0), max(value - 1, 0) * a
                assert y * e == 0
                assert (y - a) * (1 - e) == w * (1 - e)
                assert yp * (1 + a) == (1 - reset) * (y + a * out[i])
                assert 0 <= y <= 1 and 0 <= w <= 1
            predicted = 0 if reset else advance(code, pc, list(ns))
            assert pc_next == predicted
            is_pulse = int(code[pc][0] == "NOP" and code[pc][2] is not None)
            is_one = int(is_pulse and code[pc][2] == 1)
            for t in inputs:
                c, d, running, stop, s = history(h, t)
                cp, dp, rp, _, _ = history(hp, t)
                assert 0 <= c <= 1 and 0 <= d <= 1
                assert (1 - running) * stop == 0
                assert (d - Q(1, 2)) * stop == 0
                assert rp == reset + (1 - reset) * (running - stop)
                assert cp == reset * t + (1 - reset) * c * (1 + is_one * (running - stop))
                assert dp == reset * s + (1 - reset) * d * (1 + is_pulse * (running - stop))
                checks += 1
    return {"scalar_transitions": checks, "arithmetic": "exact Fractions",
            "periods": [2, 8, 32, 128, 512, 2048], "input_count": len(inputs),
            "scope": "complete controller and calibration scalar equations on these periods; no lamp source"}


def next_pulse_index(ell):
    if ell == 0:
        return 0
    # Least integer J with J^3 >= ell^4: no floating roots.
    lo, hi = 0, max(1, ell ** 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if mid ** 3 >= ell ** 4:
            hi = mid
        else:
            lo = mid + 1
    return lo


def count_and_envelope_checks():
    previous = 0
    for j in range(1, 200001):
        k = root4(j ** 3)
        assert k - previous in (0, 1)
        assert k ** 4 <= j ** 3 < (k + 1) ** 4
        previous = k
    envelopes = []
    # Q_r=log2 envelope constant <= 27r^4/256+r-1, by Legendre duality.
    for r in range(1, 25):
        max_exponent, argmax = -10 ** 9, None
        for ell in range(max(100, r ** 3)):
            p = next_pulse_index(ell)
            exponent = r * (ell + 1) - p - 1
            if exponent > max_exponent:
                max_exponent, argmax = exponent, ell
        bound = Q(27 * r ** 4, 256) + r - 1
        assert max_exponent <= bound
        envelopes.append({"order": r, "observed_log2_constant": max_exponent,
                          "maximizing_ell": argmax, "analytic_log2_bound": float(bound)})
    table = []
    for n in (2 ** 12, 2 ** 16, 2 ** 20, 2 ** 24, 2 ** 28, 2 ** 32):
        k = root4((n // 2) ** 3)
        table.append({"n": n, "k_n_over_2": k,
                      "2k_over_n": 2 * k / n,
                      "2k_over_log2_n": 2 * k / math.log2(n)})
    return {"exact_count_prefix": 200000, "envelopes": envelopes,
            "lower_bound_leading_term_table": table,
            "warning": "table omits unknown imported constants and does not evaluate optimized D_G"}


def nonuniform_moment_negative_control():
    """Every fixed moment rate can hold without a uniform growing-order rate.

    T_N=1 with probability exp(-(log N) log log log N), zero otherwise.
    For every fixed r, E T_N^(2r) <= N^-r eventually; nevertheless a
    zero-density penalty has s(1)=1/2 and sees this entire rare tail.
    Represent the parameters in logarithms, without huge allocations.
    """
    rows = []
    for x in (16.0, 64.0, 256.0, 1024.0, 4096.0):
        b = math.log(math.log(x))  # x=log N, b -> infinity.
        rows.append({"log_N": x, "moment_decay_power": b,
                     "log_expected_penalty_squared": -x * b - math.log(4),
                     "orders_certified_eventually_at_this_point": math.floor(b)})
    return {"rows": rows,
            "meaning": "fixed-order assumptions alone cannot justify choosing a growing moment order; this is an abstract spectral countermodel, not a game"}


def near_exponential_program():
    names = ["m", "length", "remaining", "countdown", "tmp", "old_length"]
    ids = {name: i for i, name in enumerate(names)}
    b = Builder()
    b.inc(ids["m"])
    b.inc(ids["length"])
    b.inc(ids["length"])
    b.mark("stage")
    b.add(ids["length"], ids["remaining"], ids["tmp"], "copy_length")
    b.clear(ids["countdown"], "clear_countdown")
    b.mark("loop")
    b.dec(ids["remaining"], "end_stage")
    b.dec(ids["countdown"], "one", "zero")
    b.mark("one")
    b.add(ids["m"], ids["countdown"], ids["tmp"], "reload_countdown")
    b.dec(ids["countdown"], "impossible", "output_one")
    b.mark("output_one")
    b.nop("loop", bit=1)
    b.mark("zero")
    b.nop("loop", bit=0)
    b.mark("end_stage")
    b.inc(ids["m"])
    b.add(ids["length"], ids["old_length"], ids["tmp"], "old_length")
    b.add(ids["old_length"], ids["length"], ids["tmp"], "double_length")
    b.clear(ids["old_length"], "clear_old_length")
    b.nop("stage")
    b.mark("impossible")
    b.nop("impossible")
    return names, b.resolve(), b.labels


def near_exponential_check(pulses=100000):
    names, code, labels = near_exponential_program()
    pc, counters, steps, total = 0, [0] * len(names), 0, 0
    stage, length, position = 1, 2, 0
    table = []
    for j in range(1, pulses + 1):
        while True:
            ins = code[pc]
            if ins[0] == "NOP" and ins[2] is not None:
                break
            pc = advance(code, pc, counters)
            steps += 1
        expected = int(position % stage == 0)
        assert ins[2] == expected, (j, stage, position, ins)
        total += ins[2]
        pc = advance(code, pc, counters)
        steps += 1
        assert max(counters) <= steps
        position += 1
        if j in (100, 1000, 10000, pulses):
            table.append({"j": j, "ones": total, "density": total / j,
                          "k_times_log2_j_over_j": total * math.log2(j) / j,
                          "instructions": steps, "instructions_per_pulse": steps / j})
        if position == length:
            stage += 1
            length *= 2
            position = 0
    (OUT / "near_exponential_program.json").write_text(json.dumps({
        "registers": names, "labels": labels, "instructions": code}, indent=2) + "\n", encoding="utf-8")
    return {"pulses": pulses, "instructions": steps, "controls": len(code),
            "registers": len(names), "table": table}


def main():
    started = time.monotonic()
    code, labels = make_program()
    # Retain only the first 8193 scalar states; never build dense clock matrices.
    trace, counter = counter_check(code)
    history = exact_history_checks(code, trace)
    counts = count_and_envelope_checks()
    near = near_exponential_check()
    report = {"status": "PASS", "counter_program": counter,
              "histories": history, "counts_and_envelopes": counts,
              "near_exponential_program": near,
              "negative_control": nonuniform_moment_negative_control(),
              "elapsed_seconds": time.monotonic() - started,
              "single_thread": True, "dependencies": "Python standard library only"}
    (OUT / "numerics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "sparse_program.json").write_text(json.dumps({"registers": REGS,
        "labels": labels, "instructions": code}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "counter": counter,
                      "history": history, "elapsed_seconds": report["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
