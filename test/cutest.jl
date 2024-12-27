using CUTEst
include("../src/nonlinearlstr.jl")
using PRIMA
using PythonCall
using NonlinearSolve

# Load the CUTEst problem
available_problems = list_sif_problems()