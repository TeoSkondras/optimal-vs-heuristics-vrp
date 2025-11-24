# main.jl – example driver using the modular VRP code.
#
# Assumes your data files live in a folder called "A" next to this script, e.g.:
#   ./A/A-n32-k5.vrp, ./A/A-n32-k5.sol, ...
#
# Usage (from project root):
#   julia main.jl

using JuMP, Gurobi

include("vrplib.jl")
include("utils_routes.jl")
include("sol_io.jl")
include("solver_scf.jl")
include("batch.jl")

using .VRPLIB
using .VRPUtils
using .SolIO
using .SolverSCF
using .BatchCVRP

base = @__DIR__

# 1) Solve a single instance as a smoke test
# Use the script folder as the anchor so it works no matter your working directory.
# vrp_path = joinpath(base, "A", "A-n32-k5.vrp")
# inst = VRPLIB.parse_vrplib(vrp_path)
# K = SolIO.parse_K_from_filename(vrp_path)
# res = SolverSCF.solve_cvrp_scf(inst; K=K, timelimit=2.0*inst.n, mipgap=0.0, seed=0)

# println("Single instance: ", vrp_path)
# println("  objective = ", res.objective,
#        " | gap = ", res.mip_gap,
#        " | runtime = ", res.runtime_sec,
#        " | vehicles = ", res.vehicles)
#
#routes = VRPUtils.extract_routes_from_x(res.xvals, inst.depot)
#VRPUtils.pretty_print_routes(routes, inst)

# 2) Batch all instances in folder "A" and write CSV
folder = joinpath(base, "A")
rows = BatchCVRP.batch_solve(folder; time_coeff=2.0, seed=0)
outcsv = joinpath(folder, "batch_results_2n_0.25h.csv")
BatchCVRP.write_results_csv(outcsv, rows)
BatchCVRP.print_results(rows)
println("\nBatch summary written to: ", outcsv)
