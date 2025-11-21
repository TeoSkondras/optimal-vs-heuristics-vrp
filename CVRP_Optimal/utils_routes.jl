module VRPUtils

using ..VRPLIB: VRPInstance

export extract_routes_from_x, route_cost, route_demand, route_total_cost,
       x_total_cost, pretty_print_routes, check_capacity, check_coverage

function extract_routes_from_x(xmat::AbstractMatrix{<:Real}, depot::Int)
    n = size(xmat, 1)
    routes = Vector{Vector{Int}}()
    starts = [j for j in 1:n if j != depot && xmat[depot, j] > 0.5]
    succ = Dict{Int,Int}()
    for i in 1:n
        bestj, bestv = 0, 0.0
        for j in 1:n
            i == j && continue
            v = xmat[i, j]
            if v > bestv
                bestv = v; bestj = j
            end
        end
        if bestv > 0.5; succ[i] = bestj; end
    end
    for s in starts
        route = Int[depot, s]; visited = Set([depot, s]); cur = s
        while true
            nxt = get(succ, cur, depot)
            push!(route, nxt)
            if nxt == depot; break; end
            if nxt in visited; push!(route, depot); break; end
            push!(visited, nxt); cur = nxt
        end
        push!(routes, route)
    end
    routes
end

route_cost(route::Vector{Int}, inst::VRPInstance) =
    sum(inst.dist[route[k], route[k+1]] for k in 1:length(route)-1)

route_demand(route::Vector{Int}, inst::VRPInstance) =
    sum(inst.demand[route[k]] for k in 2:length(route)-1)

route_total_cost(routes::Vector{Vector{Int}}, inst::VRPInstance) =
    sum(route_cost(r, inst) for r in routes)

function x_total_cost(xmat::AbstractMatrix{<:Real}, inst::VRPInstance)
    n = size(xmat, 1); c = 0.0
    for i in 1:n, j in 1:n
        if i != j && xmat[i,j] > 0.5; c += inst.dist[i,j]; end
    end
    c
end

function pretty_print_routes(routes::Vector{Vector{Int}}, inst::VRPInstance)
    if isempty(routes)
        println("No routes extracted (routes is empty)."); return
    end
    println("Vehicles: ", length(routes))
    for (i, r) in enumerate(routes)
        println("Route #", i, ": ", join(r, " -> "),
                " | demand=", route_demand(r, inst),
                " | cost=", route_cost(r, inst))
    end
    println("Total cost = ", route_total_cost(routes, inst))
end

function check_capacity(routes::Vector{Vector{Int}}, inst::VRPInstance)
    [(i, route_demand(r, inst)) for (i, r) in enumerate(routes)
        if route_demand(r, inst) > inst.capacity]
end

function check_coverage(routes::Vector{Vector{Int}}, inst::VRPInstance)
    n = inst.n; depot = inst.depot
    custs = setdiff(collect(1:n), [depot])
    seen = Int[]
    for r in routes; append!(seen, r[2:end-1]); end
    missing = setdiff(custs, seen)
    dup = length(seen) - length(unique(seen))
    missing, dup
end

end # module