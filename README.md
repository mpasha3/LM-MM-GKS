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
**A Provably Convergent MM-GKS Variant for Large-Scale Inverse Problems**
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

### Data

| File | Used by |
|---|---|
| `HSTgray.jpg` | Image deblurring (Hubble Space Telescope image) |

CT and PAT test data are generated synthetically by `PRtomo` (AIR Tools II) and `generate_PAT_June17.m`.

## Disclaimer

LM-MM-GKS is made available for research and educational purposes. The software is offered "as-is" without warranty. We welcome comments, suggestions, and contributions from users.

## Contact

- Mirjeta Pasha — [mpasha@vt.edu](mailto:mpasha@vt.edu) | [website](https://sites.google.com/view/mirjeta-pasha/home)
- Eric de Sturler — [sturler@vt.edu](mailto:sturler@vt.edu)
- Misha E. Kilmer — [misha.kilmer@tufts.edu](mailto:misha.kilmer@tufts.edu)

## Remark
The code for this package has evolved to several versions from the initial version dating back in 2022. Claude code was minimally used to reorganize some of the functions and add visualization functionalities.

## Acknowledgments

The work of Mirjeta Pasha is supported by the NSF under awards No. 2202846 and DMS 2410699. MP further acknowledges partial support from the NSF-AWM Mentoring Travel and the Isaac Newton Institute (INI) for Mathematical Sciences, Cambridge, for hospitality during the programme "Rich and Nonlinear Tomography -- a multidisciplinary approach" where partial work on this project was undertaken. The work by Eric de Sturler is based upon work supported by the National Science Foundation under Award No. 2208470. Misha Kilmer's work is partially supported by NSF HDR grant CCF-1934553 and NSF DMS-2410698. MK would like to acknowledge the Turner-Kirk Charitable Trust for the support provided by a Kirk Distinguished Visiting Fellowship to attend the aforementioned INI programme.
