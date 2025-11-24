module BatchCVRP

export batch_solve, write_results_csv, print_results

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

        row = (
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
        )

        # Print immediately for this instance
        stated = row.stated_cost === nothing ? "n/a" : string(row.stated_cost)
        diff   = row.diff === nothing ? "n/a" : string(round(row.diff; digits=2))
        diffpct = row.diff_pct === nothing ? "n/a" : string(round(row.diff_pct; digits=2)) * "%"
        println("\nInstance " * row.name *
                " | obj=" * string(row.objective) *
                " | gap=" * string(row.mip_gap) *
                " | time=" * string(row.runtime) * "s" *
                " | veh=" * string(row.vehicles) *
                " | stated=" * stated *
                " | diff=" * diff *
                " | diff%=" * diffpct)
        flush(stdout)

        push!(results, row)
    end
    return results
end

function print_results(rows)
    println("\nBatch results:")
    for r in rows
        stated = r.stated_cost === nothing ? "n/a" : string(r.stated_cost)
        diff   = r.diff === nothing ? "n/a" : string(round(r.diff; digits=2))
        diffpct = r.diff_pct === nothing ? "n/a" : string(round(r.diff_pct; digits=2)) * "%"
        println("  " * r.name *
                " | obj=" * string(r.objective) *
                " | gap=" * string(r.mip_gap) *
                " | time=" * string(r.runtime) * "s" *
                " | veh=" * string(r.vehicles) *
                " | stated=" * stated *
                " | diff=" * diff *
                " | diff%=" * diffpct)
    end
    return nothing
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
