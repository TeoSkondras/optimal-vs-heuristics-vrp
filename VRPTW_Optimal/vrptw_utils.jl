module VRPTWUtils

export routes_from_x, pretty_print_routes

"""
    routes_from_x(x, depot)

Reconstruct routes from a binary/relaxed arc matrix x by following successors from the depot.
"""
function routes_from_x(x, depot::Int)
    succ = Dict{Int,Int}()
    starts = Int[]
    for i in 1:size(x,1), j in 1:size(x,2)
        if i != j && x[i,j] > 0.5
            if i == depot
                push!(starts, j)
            else
                succ[i] = j
            end
        end
    end
    routes = Vector{Vector{Int}}()
    for s in starts
        r = [depot, s]
        cur = s
        visited = Set([depot])
        while cur != depot && haskey(succ, cur) && !(cur in visited)
            push!(visited, cur)
            nxt = succ[cur]
            push!(r, nxt)
            cur = nxt
        end
        if r[end] != depot
            push!(r, depot)
        end
        push!(routes, r)
    end
    return routes
end

function pretty_print_routes(routes)
    println("Routes:")
    for (k, r) in enumerate(routes)
        println("  Route " * string(k) * ": " * string(r))
    end
end

end # module
