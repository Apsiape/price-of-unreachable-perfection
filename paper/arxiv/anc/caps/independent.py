"""Independent finite checks; no imports from the supplied checking programs.

Run with python -u independent_checks.py. Single thread; small matrices only.
These are finite checks of the displayed algebra, not proofs of asymptotics.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import ctypes
import json
import math
import time
from fractions import Fraction as F
from pathlib import Path
import numpy as np

START = time.perf_counter()
OUT = {}
rng = np.random.default_rng(20261028)


def hs(a):
    return float(np.linalg.norm(a, 'fro'))


def unitary(d):
    q, r = np.linalg.qr(rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d)))
    return q


def positive(d):
    q = unitary(d)
    return (q*rng.random(d)) @ q.conj().T


def projection(d, rank):
    q = unitary(d)[:, :rank]
    return q @ q.conj().T


def collision_checks():
    # Exhaustive independently enumerated affine families, plus a closed formula.
    cases = pairs = 0
    for p in (2, 3, 5, 7, 11, 13):
        for k in range(1, p+1):
            q, r = divmod(p, k)
            predicted = F(k*q*(q-1)+2*r*q, p*(p-1))
            assert predicted <= F(1, k)
            for u in range(p):
                for v in range(u):
                    count = sum(((a*u+b) % p) % k == ((a*v+b) % p) % k
                                for a in range(1, p) for b in range(p))
                    assert F(count, p*(p-1)) == predicted
                    pairs += 1
            cases += 1
    OUT['affine_collisions'] = {'parameter_cases': cases, 'lamp_pairs': pairs,
                               'exact_formula_and_bound_1_over_K': True}


def hash_parameter_checks():
    cases = 0
    max_ratio = 0.0
    for ell in (1, 4, 22, 30, 34):
        for words in (1, 7, 28, 100):
            for den in (2, 7, 16, 100, 1000, 10000):
                nu = F(1, den)
                m = math.ceil(4*ell/nu)
                target = math.ceil(4*ell*ell*words*words/nu)
                k = (target-1).bit_length()
                boundary = 1-F((2*m-2*ell+1)**2, (2*m+1)**2)
                union_collision = F((ell*words)*(ell*words-1), 2*(2**k))
                assert boundary <= nu/2
                assert union_collision <= nu/8
                assert boundary+union_collision <= nu
                log_dim = 1+2*math.log2(2*m+1)+k*(6*m+10)+(2**k)
                ratio = log_dim/(float(1/nu)*math.log2(2/float(nu)))
                max_ratio = max(max_ratio, ratio)
                cases += 1
    OUT['common_bad_set_parameters'] = {'cases': cases, 'all_fraction_bounds': True,
                                       'max_dimension_ratio_fixed_grid': max_ratio}


def invariant_neighborhood_checks():
    cases = 0
    for d in (7, 11, 17, 23):
        s = np.roll(np.eye(d), 1, axis=0)
        actual = {0, 2}
        r = np.diag([-1 if i in actual else 1 for i in range(d)])
        t = (np.eye(d)-r)/2
        n1 = {(b+j) % d for b in actual for j in (-1, 0, 1)}
        n2 = {(b+j) % d for b in actual for j in (-2, -1, 0, 1, 2)}
        k = np.diag([int(i in n1) for i in range(d)])
        y = k-t
        assert np.linalg.eigvalsh(y).min() >= -1e-14
        assert np.linalg.eigvalsh(y).max() <= 1+1e-14
        pi2 = np.diag([int(i in n2) for i in range(d)])
        comm = k@s-s@k
        assert hs(comm@(np.eye(d)-pi2)) < 1e-13
        conjugate_bad = set()
        for j in range(-2, 3):
            v = np.linalg.matrix_power(s, j)
            conj = v.conj().T@r@v
            conjugate_bad.update(i for i in range(d) if abs(conj[i, i]-1)>0.5)
        assert conjugate_bad == n2
        cases += 1
    # Negative control: a chosen bad set need not be an ACTUAL failure set.
    # With every relator equal to I, conjugates fail nowhere, even if one labels
    # a block 'bad'. This refutes the proof's implication for padded bad sets.
    padded = {0}
    padded_n2 = {(b+j) % 11 for b in padded for j in range(-2, 3)}
    assert len(padded_n2) == 5
    OUT['actual_support_invariance'] = {'cases': cases, 'actual_support_conjugacy': True,
        'negative_control_padded_bad_set_has_no_failure_witness': True}


def column_checks():
    max_gap = 0.0
    max_complete = 0.0
    cases = 0
    for d in (3, 5, 8):
        eye = np.eye(d)
        for jmax in (2, 4, 7):
            for m, n in ((3, 2), (4, 2), (9, 3), (3, 3)):
                for trial in range(8):
                    phi = rng.normal(size=d)+1j*rng.normal(size=d)
                    phi /= np.linalg.norm(phi)
                    t, s = positive(d), positive(d)
                    keep = eye.astype(complex)
                    columns = []
                    cs = []
                    for j in range(jmax):
                        p = projection(d, max(1, d//2))
                        columns.append(p@keep)
                        keep = (eye-p)@keep
                        cs.append(positive(d))
                    c_end = positive(d)
                    complete = keep.conj().T@keep
                    for l in columns:
                        complete += l.conj().T@l
                    max_complete = max(max_complete, hs(complete-eye))
                    residuals = []
                    for j, (l, c) in enumerate(zip(columns, cs)):
                        residuals += [np.linalg.norm(l@t@phi-m**(-j)*c@l@phi),
                                      np.linalg.norm(l@s@phi-n**(-j-1)*l@phi)]
                    residuals.append(np.linalg.norm(keep@t@phi-m**(-jmax)*c_end@keep@phi))
                    r = max(residuals)
                    upper = m**(-jmax)+n*np.linalg.norm(s@phi)+((n+1)*math.sqrt(jmax)+1)*r
                    gap = np.linalg.norm(t@phi)-upper
                    max_gap = max(max_gap, gap)
                    assert gap <= 1e-11
                    cases += 1
    OUT['noncommuting_complete_columns_m_ge_n'] = {'cases': cases,
        'max_completeness_error': max_complete, 'max_positive_bound_violation': max_gap}


def sequential_povm_checks():
    max_normalization = max_negative = max_ratio = max_comm_ratio = 0.0
    cases = 0
    for d, answers in ((3, 3), (4, 4), (6, 3)):
        eye = np.eye(d)
        partition = [np.diag([int(j % answers == a) for j in range(d)]).astype(complex)
                     for a in range(answers)]
        for theta in (0.0, 1e-2, 1e-3, 1e-4, 1e-5):
            for trial in range(7):
                ps = []
                for a, p in enumerate(partition):
                    h = rng.normal(size=(d,d))+1j*rng.normal(size=(d,d))
                    h = (h+h.conj().T)/2
                    vals, vecs = np.linalg.eigh(h)
                    u = (vecs*np.exp(1j*theta*vals))@vecs.conj().T
                    ps.append(u@p@u.conj().T)
                weights = 1+theta*rng.random(d)
                weights /= weights.sum()
                xi = np.diag(np.sqrt(weights))
                eta = hs((sum(ps)-eye)@xi)
                for a in range(answers):
                    eta = max(eta, hs(ps[a]@xi-xi@ps[a]))
                    for b in range(a):
                        eta = max(eta, hs(ps[a]@ps[b]@xi), hs(ps[b]@ps[a]@xi))
                keep = eye.astype(complex)
                effects = []
                for a in range(answers-1):
                    effects.append(keep.conj().T@ps[a]@keep)
                    keep = (eye-ps[a])@keep
                effects.append(keep.conj().T@keep)
                max_normalization = max(max_normalization, hs(sum(effects)-eye))
                for a, effect in enumerate(effects):
                    max_negative = max(max_negative, -float(np.linalg.eigvalsh(effect).min()))
                    if eta > 1e-12:
                        max_ratio = max(max_ratio, hs((effect-ps[a])@xi)/eta)
                        max_comm_ratio = max(max_comm_ratio, hs(effect@xi-xi@effect)/eta)
                assert hs(sum(effects)-eye) < 1e-11
                assert max_negative < 1e-11
                cases += 1
    OUT['weighted_sequential_POVM_rounding'] = {'cases': cases,
        'max_normalization_error': max_normalization, 'max_negative_eigenvalue': max_negative,
        'max_effect_error_over_input_precision': max_ratio,
        'max_effect_commutator_over_input_precision': max_comm_ratio}



for check in (collision_checks, hash_parameter_checks, invariant_neighborhood_checks,
              column_checks, sequential_povm_checks):
    check()
    print(check.__name__, 'PASS', flush=True)
OUT['elapsed_seconds'] = time.perf_counter()-START
if os.name == 'nt':
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)]+[
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize',
            'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        OUT['peak_working_set_bytes'] = counters.PeakWorkingSetSize
Path(__file__).with_suffix('.json').write_text(json.dumps(OUT, indent=2), encoding='utf-8')
print(json.dumps(OUT, indent=2))
print('ALL INDEPENDENT CHECKS PASSED', flush=True)
