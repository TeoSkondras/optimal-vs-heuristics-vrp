# main.jl – VRPTW solver driver (Solomon format)

using JuMP
using Gurobi

include("vrptw_parser.jl")
include("solver_vrptw.jl")
include("vrptw_utils.jl")
include("batch_vrptw.jl")

using .VRPTWLIB: VRPTWInstance, parse_solomon, n_customers
using .SolverVRPTW
using .VRPTWUtils
using .BatchVRPTW

# Path to an instance in the In/ folder
base = @__DIR__
vrp_path = joinpath(base, "In", "c101.txt")

inst = parse_solomon(vrp_path)
println("Loaded instance ", inst.name, ": n=", n_customers(inst), " capacity=", inst.capacity, " vehicles=", inst.vehicles)

res = Base.invokelatest(SolverVRPTW.solve_vrptw, inst; timelimit=60.0)
println("Objective=", res.objective, " | time=", res.runtime_sec, "s | vehicles=", res.vehicles_used, " | term=", res.term_status)

routes = VRPTWUtils.routes_from_x(res.xvals, inst.depot)
VRPTWUtils.pretty_print_routes(routes)

# Batch example (set `do_batch` to true to run all instances)
do_batch = true
if do_batch
    folder = joinpath(base, "In")
    rows = BatchVRPTW.batch_solve(folder; timelimit_coeff=10.0)
    BatchVRPTW.write_results_csv(joinpath(base, "batch_results_10n.csv"), rows)
end
