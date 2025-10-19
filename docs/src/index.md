# nonlinearlstr.jl

Documentation for [nonlinearlstr.jl](https://github.com/vcantarella/nonlinearlstr).

## Overview

nonlinearlstr.jl is a Julia package for nonlinear least squares optimization using trust region methods.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/vcantarella/nonlinearlstr")
```

## Quick Start

```julia
using nonlinearlstr

# Define your residual function and Jacobian
function residual(x)
    return [x[1]^2 + x[2]^2 - 1, x[1] - x[2]]
end

function jacobian(x) 
    return [2*x[1] 2*x[2]; 1 -1]
end

# Initial guess
x0 = [0.5, 0.5]

# Solve using trust region method
result = lm_trust_region(residual, jacobian, x0)
```

## Features

- Trust region methods with QR and SVD factorization strategies
- Scaling strategies for better conditioning
- Bounded optimization support
- Comprehensive testing and benchmarking