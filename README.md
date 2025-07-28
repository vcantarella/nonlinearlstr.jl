# nonlinearlstr

[![Build Status](https://github.com/vcantarella/nonlinearlstr.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/vcantarella/nonlinearlstr.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Overview
`nonlinearlstr.jl` is a simple Julia package for solving nonlinear least squares problems using (primarily) trust region methods. It is supposed to test implementations and pick robust ones. -> in development.
I don't envision this as a registered package, but rather a collection of useful algorithms that can be better optimized and used in other packages.

## Features

### Core Algorithms
- **QR-based Trust Region**: Numerically stable trust region solver using QR decomposition. The subproblem is solved using a regularized least squares approach based on the Levenberg-Marquardt algorithm.
- **TCG Trust Region**: Classical trust region method for nonlinear least squares where the subproblem is solved using Truncated Conjugate Gradient (TCG) algorithm
- **General TCG Trust Region**: Trust region solver for general optimization problems with explicit cost/gradient/Hessian where the subproblem is solved using TCG

### Specialized Features
- **Levenberg-Marquardt**: Classical LM algorithm with adaptive regularization parameter finding - needs to better accounts for bounds
- **Truncated Conjugate Gradient (TCG)**: subproblem solver for bounds and large-scale problems - Needs better evaluation and necessity of affine scaling. Maybe it can be improved with the TRSBOX algorithm from Powell's BOBYQA solver.
- **Affine Scaling**: Enhanced convergence for bound-constrained problems - Needs better evaluation to check if it is worth it.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/vcantarella/nonlinearlstr.jl")
```
