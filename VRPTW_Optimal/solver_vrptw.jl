module SolverVRPTW

export solve_vrptw

using JuMP
using JuMP: set_time_limit_sec, objective_value, value, result_count, solve_time
using LinearAlgebra
using Gurobi
import MathOptInterface as MOI
using ..VRPTWLIB: VRPTWInstance, n_customers, euclidean_distance

struct VRPTWResult
    objective::Float64
    runtime_sec::Float64
    mip_gap::Float64
    vehicles_used::Int
    term_status::MOI.TerminationStatusCode
    xvals::Matrix{Float64}
    times::Vector{Float64}
end

function solve_vrptw(inst::VRPTWInstance; timelimit::Float64=600.0)
    n = n_customers(inst)
    nodes = 1:(n + 1)
    customers = 2:(n + 1)
    depot = inst.depot

    dist = [euclidean_distance(inst, i, j) for i in nodes, j in nodes]
    max_dist = maximum(dist)
    max_service = maximum(inst.service)
    window_span = maximum(inst.due) - minimum(inst.ready)
    bigM = window_span + max_service + max_dist

    model = Model(Gurobi.Optimizer)
    set_time_limit_sec(model, timelimit)
    #set_silent(model)
    set_optimizer_attribute(model, "InfUnbdInfo", 1)

    @variable(model, x[i in nodes, j in nodes], Bin)
    @variable(model, 0 <= load[i in nodes] <= inst.capacity)
    @variable(model, 0 <= t[i in nodes])
    @variable(model, 0 <= y <= inst.vehicles, Int)

    # No self loops
    for i in nodes
        fix(x[i, i], 0; force=true)
    end
    @constraint(model, [k in customers], sum(x[i, k] for i in nodes if i != k) == 1)
    @constraint(model, [k in customers], sum(x[k, j] for j in nodes if j != k) == 1)

    @constraint(model, sum(x[depot, j] for j in customers) == y)
    @constraint(model, sum(x[i, depot] for i in customers) == y)

    @constraint(model, load[depot] == 0)
    @constraint(model, [k in customers], load[k] >= inst.demand[k])
    @constraint(model, [i in nodes, j in customers; i != j], load[j] >= load[i] + inst.demand[j] - inst.capacity * (1 - x[i, j]))

    @constraint(model, t[depot] == inst.ready[depot])
    @constraint(model, [i in nodes], inst.ready[i] <= t[i] <= inst.due[i])
    @constraint(model, [i in nodes, j in customers; i != j], t[j] >= t[i] + inst.service[i] + dist[i, j] - bigM * (1 - x[i, j]))

    @objective(model, Min, sum(dist[i, j] * x[i, j] for i in nodes, j in nodes if i != j))

    optimize!(model)

    term = JuMP.termination_status(model)
    runtime = solve_time(model)

    if result_count(model) == 0
        return VRPTWResult(NaN, runtime, NaN, 0, term, zeros(length(nodes), length(nodes)), zeros(length(nodes)))
    end

    obj = objective_value(model)
    gap = try
        MOI.get(model, MOI.RelativeGap())
    catch
        NaN
    end
    xval = value.(x)
    tval = value.(t)
    vehicles_used = round(Int, sum(xval[depot, j] for j in customers))

    return VRPTWResult(obj, runtime, gap, vehicles_used, term, xval, tval)
end

end # module
