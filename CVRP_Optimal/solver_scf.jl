module SolverSCF

using JuMP, Gurobi
import MathOptInterface as MOI
using ..VRPLIB: VRPInstance
using ..VRPUtils: extract_routes_from_x

export CVRPResult, solve_cvrp_scf

mutable struct CVRPResult
    objective::Float64
    routes::Vector{Vector{Int}}
    xvals::Matrix{Float64}
    runtime_sec::Float64
    mip_gap::Float64
    vehicles::Int
    term_status::MOI.TerminationStatusCode
    primal_status::MOI.ResultStatusCode
end

function solve_cvrp_scf(inst::VRPInstance; K::Union{Nothing,Int}=nothing,
                        timelimit::Float64=180.0, mipgap::Float64=0.0,
                        seed::Int=0)
    n = inst.n; depot = inst.depot
    V = 1:n; customers = [i for i in V if i != depot]; Q = inst.capacity

    model = Model(Gurobi.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "TimeLimit", timelimit)
    set_optimizer_attribute(model, "MIPGap", mipgap)
    set_optimizer_attribute(model, "MIPFocus", 3)
    set_optimizer_attribute(model, "Cuts", 2)
    #set_optimizer_attribute(model, "Heuristics", 0.0) # disable heuristics to get more stable results, but can lead to worse solutions in some cases
    #set_optimizer_attribute(model, "Heuristics", 0.25) # more aggressive heuristics, but can lead to worse solutions in some cases
    set_optimizer_attribute(model, "Presolve", 2)
    set_optimizer_attribute(model, "Seed", seed)

    @variable(model, x[i in V, j in V; i != j], Bin)
    @variable(model, f[i in V, j in V; i != j] >= 0)

    @constraint(model, [i in customers], sum(x[i,j] for j in V if j != i) == 1)
    @constraint(model, [j in customers], sum(x[i,j] for i in V if i != j) == 1)

    @constraint(model, sum(x[depot,j] for j in V if j != depot) ==
                         sum(x[i,depot] for i in V if i != depot))
    if K !== nothing
        @constraint(model, sum(x[depot,j] for j in V if j != depot) <= K)
    end

    b = zeros(Float64, n)
    b[depot] = -sum(inst.demand[i] for i in customers)
    for i in customers; b[i] = inst.demand[i]; end
    @constraint(model, [v in V],
        sum(f[i,v] for i in V if i != v) - sum(f[v,j] for j in V if j != v) == b[v])

    @constraint(model, [i in V, j in V; i != j], f[i,j] <= Q * x[i,j])
    @constraint(model, [i in customers, j in customers; i < j], x[i,j] + x[j,i] <= 1)

    @objective(model, Min, sum(inst.dist[i,j]*x[i,j] for i in V for j in V if i != j))

    runtime = @elapsed optimize!(model)

    term = termination_status(model)
    prim = primal_status(model)
    rc = MOI.get(model, MOI.ResultCount())
    obj = rc > 0 ? objective_value(model) : NaN

    xmat = zeros(Float64, n, n)
    if rc > 0
        for i in V, j in V
            if i != j
                xv = value(x[i,j]); xmat[i,j] = isnan(xv) ? 0.0 : xv
            end
        end
    end

    routes = rc > 0 ? extract_routes_from_x(xmat, depot) : Vector{Vector{Int}}()
    vehs = rc > 0 ? length(routes) : 0
    gap = rc > 0 ? try MOI.get(model, MOI.RelativeGap()) catch; NaN end : NaN

    return CVRPResult(obj, routes, xmat, runtime, gap, vehs, term, prim)
end

end # module