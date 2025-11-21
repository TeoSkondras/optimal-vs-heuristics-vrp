module BatchCVRP

export batch_solve, write_results_csv

using ..VRPLIB: VRPInstance, parse_vrplib
using ..SolverSCF: solve_cvrp_scf
using ..SolIO: parse_K_from_filename, stated_cost_from_sol, basename_safe

function batch_solve(folder::AbstractString; time_coeff::Float64=10.0, seed::Int=0)
    @assert isdir(folder) "Not a directory: $folder"
    vrp_files = sort(filter(f -> endswith(lowercase(f), ".vrp"),
                            readdir(folder; join=true)))
    results = NamedTuple[]
    for (k, vrp_path) in enumerate(vrp_files)
        name = basename_safe(vrp_path)
        inst = parse_vrplib(vrp_path)
        tl   = time_coeff * inst.n
        K    = parse_K_from_filename(vrp_path)

        res  = solve_cvrp_scf(inst; K=K, timelimit=tl, mipgap=0.0, seed=seed)

        sol_path = replace(vrp_path, ".vrp" => ".sol")
        sc = isfile(sol_path) ? stated_cost_from_sol(sol_path) : nothing

        push!(results, (
            name = name,
            n = inst.n,
            K = K,
            capacity = inst.capacity,
            timelimit = tl,
            objective = res.objective,
            mip_gap = res.mip_gap,
            runtime = res.runtime_sec,
            vehicles = res.vehicles,
            term = string(res.term_status),
            primal = string(res.primal_status),
            stated_cost = sc,
            diff = (sc === nothing || isnan(res.objective)) ? nothing : (res.objective - sc),
            diff_pct = (sc === nothing || isnan(res.objective)) ? nothing : (100 * (res.objective - sc) / sc),
        ))
    end
    return results
end

function write_results_csv(path::AbstractString, rows)
    open(path, "w") do io
        println(io, "name,n,K,capacity,timelimit,objective,mip_gap,runtime,vehicles,term,primal,stated_cost,diff,diff_pct")
        for r in rows
            println(io, string(
                r.name, ",", r.n, ",", (r.K === nothing ? "" : r.K), ",", r.capacity, ",",
                r.timelimit, ",", r.objective, ",", r.mip_gap, ",", r.runtime, ",",
                r.vehicles, ",", r.term, ",", r.primal, ",",
                (r.stated_cost === nothing ? "" : r.stated_cost), ",",
                (r.diff === nothing ? "" : r.diff), ",",
                (r.diff_pct === nothing ? "" : r.diff_pct)
            ))
        end
    end
    return path
end

end # module