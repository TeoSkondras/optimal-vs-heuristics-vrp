module BatchVRPTW

export batch_solve, write_results_csv

using ..VRPTWLIB: parse_solomon, n_customers
using ..SolverVRPTW: solve_vrptw

function batch_solve(folder::AbstractString; timelimit_coeff::Float64=10.0, seed::Int=0)
    @assert isdir(folder) "Not a directory: $folder"
    vrp_files = sort(filter(f -> endswith(lowercase(f), ".txt"), readdir(folder; join=true)))
    results = NamedTuple[]
    for vrp_path in vrp_files
        inst = parse_solomon(vrp_path)
        tl = timelimit_coeff * n_customers(inst)
        res = solve_vrptw(inst; timelimit=tl)
        row = (
            name = inst.name,
            n = n_customers(inst),
            K = nothing,
            capacity = inst.capacity,
            timelimit = tl,
            objective = res.objective,
            mip_gap = res.mip_gap,
            runtime = res.runtime_sec,
            vehicles = res.vehicles_used,
            term = string(res.term_status),
            primal = "",
            stated_cost = nothing,
            diff = nothing,
            diff_pct = nothing,
        )
        println("Instance " * row.name *
                " | obj=" * string(row.objective) *
                " | gap=" * string(row.mip_gap) *
                " | time=" * string(row.runtime) * "s" *
                " | veh=" * string(row.vehicles) *
                " | term=" * row.term)
        push!(results, row)
    end
    return results
end

function write_results_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "name,n,K,capacity,timelimit,objective,mip_gap,runtime,vehicles,term,primal,stated_cost,diff,diff_pct")
        for r in rows
            println(io, string(
                r.name, ",", r.n, ",",
                "", ",", r.capacity, ",",
                r.timelimit, ",", r.objective, ",", r.mip_gap, ",", r.runtime, ",",
                r.vehicles, ",", r.term, ",", r.primal, ",",
                "", ",", "", ",", ""
            ))
        end
    end
    return path
end

end # module
