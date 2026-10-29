# The price of unreachable perfection: dimension laws at every regular scale

Seth Douglas and Nidhal Mghirbi. Version 1.0.1, October 2026. DOI: [10.5281/zenodo.23774171](https://doi.org/10.5281/zenodo.23774171). Concept DOI (all versions): [10.5281/zenodo.23754198](https://doi.org/10.5281/zenodo.23754198). A preprint; it has not been peer reviewed.

Version 1.0.1 changes packaging only. It adds the recorded check outputs (the `*.log` files listed in the checksum manifests), which the version 1.0.0 repository omitted, stores every file byte for byte so that all SHA-256 manifests verify from a fresh clone, and refreshes one stale manifest entry. The paper (`paper/main.pdf`, dated version 1.0.0) is unchanged. Version 1.0.0: [10.5281/zenodo.23754199](https://doi.org/10.5281/zenodo.23754199). All versions: [10.5281/zenodo.23754198](https://doi.org/10.5281/zenodo.23754198).

How expensive is an optimum that a finite quantum system can approach but never attain? For a fixed finite rational nonlocal game of quantum value one, write $A_G(n)=\log_2D_G(2^{-n})$, where $D_G$ is the least local dimension at loss $2^{-n}$. We prescribe $\log_2 A_G(n)$ at every logarithmic-space, nondecreasing unit-increment scale $F$ with $F(n)/\log n\to\infty$, up to an additive $O(\log n+\log^2(F(n)+2))$ error, and match the one-way distillable entanglement of the original mixed strategy state. At every asymptotically flat such scale, the positive finite lower and upper normalized dimension endpoints are exactly the ordered pairs with lower endpoint in $\Sigma^0_2$ and upper endpoint in $\Pi^0_2$; existing positive limits are exactly $\Delta^0_2$. A single uniform spectral-tail construction supplies the density-free profile law behind these results. It also classifies stretched-scale pairs in the unit square and gives quasi-polynomial dimension windows of width three in the exponent. A computable unit-increment target outside every rational game's profile shows that computability alone is insufficient. We locate the boundaries through a quantitative locality cap, an honest-linear compression cap, and an explicitly conditional route to finite exponents above one. A separate exact-marginal correlation theorem identifies a Farey law and its Diophantine endpoint curve. The dimension arithmetic necessity applies to all rational games; the operational realization statements concern the constructed games.

## Contents
- `paper/main.pdf`, with its LaTeX source (`paper/main.tex`, `paper/sections/`, `paper/references.bib`, `paper/main.bbl`) and a builder
  (`paper/build.py`).
- `paper/arxiv/`: a standalone source folder with the same source and checks.
- `anc/`: the finite checks, with a single-threaded runner (`anc/run_checks.py`), recorded outputs and SHA-256 manifests. See
  `anc/README.md` for their coverage. The theorems rest on the written proofs; the computations support them and do not replace them.

## Building
Build the paper in a new empty directory with:

    python paper/build.py path-to-new-build-directory

The script uses ordinary `pdflatex`, `bibtex`, and three further `pdflatex` passes. It fails on any final LaTeX warning, undefined reference, overfull box or underfull box, or any BibTeX warning. The standalone directory can also be copied alone and built with the same commands directly.

## Related
The base classifications are cited from "The uncomputable price of precision" (Seth Douglas and Nidhal Mghirbi), version 4.0.0,
[10.5281/zenodo.23753741](https://doi.org/10.5281/zenodo.23753741).

## License
The paper and documentation are licensed CC BY 4.0, and the scripts MIT or Apache 2.0, at your option. See RIGHTS.md.
