# LM-MM-GKS

<p align="center">
  <b>Limited Memory Majorization-Minimization Generalized Krylov Subspace Method for Large-Scale Inverse Problems</b>
</p>

LM-MM-GKS is a MATLAB package for solving large-scale edge-preserving inverse problems with limited memory. The package implements:

1. **LM-MM-GKS**: a provably convergent variant of MM-GKS that alternately compresses and expands the search space, keeping memory requirements bounded independent of the number of iterations.
2. **s-LM-MM-GKS**: a streaming variant that handles problems where data arrives sequentially or exceeds memory capacity.
3. **Four compression strategies**: truncated SVD (tSVD), reduced basis decomposition (RBD), solution-oriented compression (SOC), and sparsity-enforcing compression (SEC).

### Problem Formulation

LM-MM-GKS solves regularized inverse problems of the form

$$\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{A}\mathbf{x} - \mathbf{d}\|_2^2 + \frac{\lambda}{q} \|\Psi \mathbf{x}\|_q^q, \quad 0 < q \leq 2,$$

where $\Psi$ is a regularization operator (e.g., discrete gradient for edge preservation) and $\lambda$ is automatically selected via GCV on the small projected problem at each iteration.

## Reference

Mirjeta Pasha, Eric de Sturler, Misha E. Kilmer,
**A Provably Convergent MM-GKS Variant for Large-Scale Inverse Problems**,
submitted to SIAM Journal on Scientific Computing (SISC), 2026.

#### Link of the paper
[arXiv preprint - link TBD]

## Requirements

- MATLAB R2020b or later (tested on R2023b)
- Image Processing Toolbox (for `ssim`, `psnr`)
- [IRTools](https://github.com/jnagy1/IRtools) (included)
- [AIR Tools II](https://github.com/jakobsj/AIRToolsII) (included)

## Building LM-MM-GKS from source code

##### Clone the repository

```matlab
git clone https://github.com/mpasha3/LM-MM-GKS
```

##### Setup paths in MATLAB

```matlab
cd LMMGKS_Aug25_2026
addpath(pwd)
addpath('RMMGKS2')
addpath('AIRToolsII-master'); AIRToolsII_setup
addpath('IRTools'); IRtools_setup
```

Alternatively, run `startup_Recycle.m` to set up all paths.

##### Run a quick test

```matlab
rev1_Deblurring_Test1_June17   % Image deblurring (Table 1, Figures 1-2)
```

## Reproducing Paper Results

| Paper Section | Result | Script |
|---|---|---|
| Section 7.1 (Image Deblurring) | Table 1, Figures 1-2 | `rev1_Deblurring_Test1_June17.m` |
| Section 7.2.1 (Streaming CT, Test 1) | Figures 3-4, Table 2 (all noise levels) | `run_all_noise_levels_June18.m` |
| Section 7.2.1 (Streaming CT, Test 1) | Figure 5 (RRE convergence) | `fig5_paper_June18.m` |
| Section 7.2.2 (Streaming CT, Test 2) | Table, figures | `rev1_Tomo_Test2_June18.m` |
| Section 7.3 (Dynamic PAT) | Figures 7-8 | `rev1_PAT_Test3_lplq_June18.m` |
| Section 7.3 (Dynamic PAT) | Table 3 (all noise levels) | `run_PAT_Table3_all_June18.m` |

## Overview of the Package Structure

### Solver Functions (LM-MM-GKS with different compression)

| Function | Compression | V0 Support |
|---|---|---|
| `grad_MMGKS_..._TSVD_June17.m` | Truncated SVD (tSVD) | Yes |
| `grad_MMGKS_..._RBD_June16.m` | Reduced Basis Decomposition (RBD) | Yes |
| `grad_MMGKS_..._sol_oriented_June17.m` | Solution-Oriented Compression (SOC) | Yes |
| `grad_MMGKS_..._sparsity_June17.m` | Sparsity-Enforcing Compression (SEC) | Yes |

All solvers accept an optional 12th argument `V0` (an initial subspace for recycling), used by the streaming variant s-LM-MM-GKS to pass compressed subspace information between subproblems.

### Streaming Solvers

| Function | Description |
|---|---|
| `RMMGKS_stream_3prob_June17.m` | s-LM-MM-GKS for 3 subproblems (fixed iterations) |
| `RMMGKS_stream_3prob_tol_June17.m` | s-LM-MM-GKS for 3 subproblems (tolerance stopping) |
| `RMMGKS_stream_6prob_June18.m` | s-LM-MM-GKS for 6 subproblems (fixed iterations) |
| `RMMGKS_stream_6prob_tol_June18.m` | s-LM-MM-GKS for 6 subproblems (tolerance stopping) |

### Baseline Methods

| Function | Description |
|---|---|
| `lplq_GCV.m` | Standard MM-GKS solver (no compression, GCV) |
| `lplq_June18.m` | Restarted MM-GKS (Buccini & Reichel, DP + adaptive majorant) |

### Utilities

| Function | Purpose |
|---|---|
| `HaarPSI.m` | Haar wavelet-based perceptual similarity index |
| `KrylFixed.m` | Golub-Kahan bidiagonalization for initial subspace |
| `solveProjTikhonovGCV.m` | GCV-based regularization parameter selection |
| `solveProjTikhonovDP.m` | Discrepancy principle for regularization parameter |
| `generate_PAT_June17.m` | Dynamic PAT test problem generator |
| `build_L1.m`, `build_L_R1a.m` | Discrete derivative regularization operators |
| `RBD.m` | Reduced basis decomposition |

### External Toolbox Directories

| Directory | Purpose |
|---|---|
| `AIRToolsII-master/` | Algebraic Iterative Reconstruction tools (tomography test problems) |
| `IRTools/` | Iterative Regularization tools |
| `HyBRrecycle/` | HyBR with subspace recycling (comparison method) |
| `lplq/` | lplq package by Buccini & Reichel |
| `RMMGKS2/` | Earlier RMMGKS code (provides `build_L.m`) |
| `@Aclass/` | Operator class for implicit matrix-vector products |

### Data

| File | Used by |
|---|---|
| `HSTgray.jpg` | Image deblurring (Hubble Space Telescope image) |

CT and PAT test data are generated synthetically by `PRtomo` (AIR Tools II) and `generate_PAT_June17.m`.

## Key Algorithmic Details

The main algorithm (Algorithm 3.3 in the paper) alternates between:

1. **Enlarge** (Algorithm 3.1): expands the search space by adding basis vectors, using the gradient of the *updated* quadratic tangent majorant. This change from standard MM-GKS is key to the convergence proof.
2. **Compress** (Algorithm 3.2): reduces the search space to $k_{\min}$ vectors while always retaining the current solution and gradient/residual.

Key parameters:
- `kmax`: maximum search space dimension (memory budget)
- `kmin` (passed as `r`): compressed dimension after each cycle
- `kiter`: number of inner expansion steps per cycle
- `iter`: number of outer expand-compress cycles
- `epsilon`: smoothing parameter for the $\ell_1$ approximation

## Disclaimer

LM-MM-GKS is made available for research and educational purposes. The software is offered "as-is" without warranty. We welcome comments, suggestions, and contributions from users.

## Contact

- Mirjeta Pasha — [mpasha@vt.edu](mailto:mpasha@vt.edu) | [website](https://sites.google.com/view/mirjeta-pasha/home)
- Eric de Sturler — [sturler@vt.edu](mailto:sturler@vt.edu)
- Misha E. Kilmer — [misha.kilmer@tufts.edu](mailto:misha.kilmer@tufts.edu)

## Acknowledgments

The work of Mirjeta Pasha is supported by the NSF under awards No. 2202846 and DMS 2410699. MP further acknowledges partial support from the NSF-AWM Mentoring Travel and the Isaac Newton Institute (INI) for Mathematical Sciences, Cambridge, for hospitality during the programme "Rich and Nonlinear Tomography -- a multidisciplinary approach" where partial work on this project was undertaken. The work by Eric de Sturler is based upon work supported by the National Science Foundation under Award No. 2208470. Misha Kilmer's work is partially supported by NSF HDR grant CCF-1934553 and NSF DMS-2410698. MK would like to acknowledge the Turner-Kirk Charitable Trust for the support provided by a Kirk Distinguished Visiting Fellowship to attend the aforementioned INI programme.
