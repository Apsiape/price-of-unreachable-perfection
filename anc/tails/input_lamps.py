"""Check 1: lamp-row residuals of the affine Clifford source depend only on c = <v_u, v_w>.

For unit real vectors v0, vw with inner product c, build Majoranas gamma(f (x) e_a) on the span
(two vectors x three copies = 6 Majoranas on 3 qubits, Jordan-Wigner), set
X_u = i A_u B_u, Z_u = i A_u C_u, and compute the normalized group residual (w - I)/2 of the four
lamp words [x, y] = x y x^-1 y^-1 with y = X_w or Z_w (rows 23-26 of the certificate table of the marked affine source).
Claims checked: mixed rows have |R| = |c| I exactly; same-type rows have ||R|| <= 2|c| sqrt(1-c^2).
Also checks J = (X_0 Z_0)^2 = -I.
Single-threaded numpy on 8x8 matrices.
"""
import numpy as np

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], dtype=complex); Y = np.array([[0, -1j], [1j, 0]]); Zp = np.diag([1, -1]).astype(complex)

def kron(*ms):
    out = np.eye(1, dtype=complex)
    for m in ms:
        out = np.kron(out, m)
    return out

def majoranas(nq):
    g = []
    for k in range(nq):
        pre = [Zp] * k; post = [I2] * (nq - k - 1)
        g.append(kron(*pre, X, *post)); g.append(kron(*pre, Y, *post))
    return g

G = majoranas(3)  # 6 Majoranas: index (i, a) -> basis f_i (x) e_a, i in {0,1}, a in {0,1,2}
def gam(f):  # f: dict (i,a)->coef, real
    return sum(coef * G[3 * i + a] for (i, a), coef in f.items())

def ops(c):
    s = np.sqrt(1 - c * c)
    # v0 = f_0 ; vw = c f_0 + s f_1
    A0 = gam({(0, 0): 1}); B0 = gam({(0, 1): 1}); majorana_c_zero = gam({(0, 2): 1})
    Aw = gam({(0, 0): c, (1, 0): s}); Bw = gam({(0, 1): c, (1, 1): s}); Cw = gam({(0, 2): c, (1, 2): s})
    X0 = 1j * A0 @ B0; Z0 = 1j * A0 @ majorana_c_zero; Xw = 1j * Aw @ Bw; Zw = 1j * Aw @ Cw
    return X0, Z0, Xw, Zw

def resid(x, y):
    w = x @ y @ np.linalg.inv(x) @ np.linalg.inv(y)
    return (w - np.eye(w.shape[0])) / 2

worst = {"mixed_dev_from_|c|I": 0.0, "same_ratio_to_2|c|sqrt(1-c^2)": 0.0, "J_plus_I": 0.0, "reflection_dev": 0.0}
for c in np.linspace(-1, 1, 41):
    X0, Z0, Xw, Zw = ops(c)
    for M in (X0, Z0, Xw, Zw):
        worst["reflection_dev"] = max(worst["reflection_dev"], np.abs(M @ M - np.eye(8)).max(), np.abs(M - M.conj().T).max())
    J = (X0 @ Z0) @ (X0 @ Z0)
    worst["J_plus_I"] = max(worst["J_plus_I"], np.abs(J + np.eye(8)).max())
    for (x, y) in ((X0, Zw), (Z0, Xw)):  # rows 24, 25 (mixed)
        R = resid(x, y)
        absR = np.sqrt(np.clip(np.linalg.eigvalsh(R.conj().T @ R), 0, None))
        worst["mixed_dev_from_|c|I"] = max(worst["mixed_dev_from_|c|I"], np.abs(absR - abs(c)).max())
    for (x, y) in ((X0, Xw), (Z0, Zw)):  # rows 23, 26 (same type)
        R = resid(x, y)
        nrm = np.linalg.norm(R, 2)
        bound = 2 * abs(c) * np.sqrt(max(1 - c * c, 0))
        if bound > 1e-12:
            worst["same_ratio_to_2|c|sqrt(1-c^2)"] = max(worst["same_ratio_to_2|c|sqrt(1-c^2)"], nrm / bound)
        else:
            worst["same_ratio_to_2|c|sqrt(1-c^2)"] = max(worst["same_ratio_to_2|c|sqrt(1-c^2)"], 1.0 if nrm < 1e-9 else np.inf)
print("lamp residual check:", {k: float(v) for k, v in worst.items()}, flush=True)
ok = worst["mixed_dev_from_|c|I"] < 1e-9 and worst["same_ratio_to_2|c|sqrt(1-c^2)"] <= 1 + 1e-9 and worst["J_plus_I"] < 1e-9 and worst["reflection_dev"] < 1e-9
print("PASS" if ok else "FAIL", flush=True)
