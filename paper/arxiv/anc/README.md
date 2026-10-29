# Finite checks

Run `python anc/run_checks.py` from the distribution root. The runner executes thirteen programs sequentially, fixes numerical thread counts to one, and applies a 1400 MiB per-process memory limit on Windows. Python 3 and NumPy are required. Console outputs are in each topic's `recorded/` directory; machine-readable outputs remain beside the programs.

Each topic has construction checks and checks written separately that re-derive identities independently. The programs cover Lemma 3.1 (tails and Clifford matrices), Theorem 4.1 and Corollaries 4.2--4.4 (profiles and sparse streams), Theorems 5.2, 5.4 and 5.5 (tracking and windows), Theorem 5.6 (integer-root envelopes), the locality and stopped-multiplier statements recalled in Section 6.1, Lemma 6.4 and Theorems 6.5--6.6 (question-pair counts, stopping costs, corner ranks and cumulative costs), and Theorem 7.1 (Farey arithmetic). The window-charge program also checks the relator/marker distinction and finite fair-cylinder identities used in the non-attainment proof.

These computations test finite identities and examples. They do not prove asymptotic statements by finite extrapolation, formalize the compiler or extraction theorem, or establish the conditional compression hypothesis. The profile-band diagnostic computes a sufficient upper envelope, not an exact worst spectral law.

`SHA256SUMS` lists every shipped ancillary file except itself, with LF line endings. It includes all programs, instruction tables, recorded logs and machine-readable results. Runtime fields and low floating-point digits can vary between systems. Checks of unstated pairwise-hash barriers and auxiliary compression claims are excluded.
