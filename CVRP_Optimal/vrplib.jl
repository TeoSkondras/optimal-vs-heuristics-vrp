module VRPLIB

export VRPInstance, parse_vrplib

struct VRPInstance
    name::String
    n::Int
    capacity::Int
    coords::Vector{Tuple{Float64,Float64}}
    demand::Vector{Int}
    depot::Int
    dist::Matrix{Float64}
end

function parse_vrplib(path::AbstractString)::VRPInstance
    name = ""; n = 0; cap = 0; edge_type = ""
    coords = Dict{Int,Tuple{Float64,Float64}}()
    demand = Dict{Int,Int}()
    depot = 1
    section = ""

    open(path, "r") do io
        for ln in eachline(io)
            s = strip(ln); isempty(s) && continue
            if occursin(":", s) && !(startswith(s, "NODE_COORD_SECTION")
               || startswith(s, "DEMAND_SECTION") || startswith(s, "DEPOT_SECTION"))
                key, val = strip.(split(s, ":", limit=2))
                ku = uppercase(key)
                if ku == "NAME"
                    name = val
                elseif ku == "DIMENSION"
                    n = parse(Int, split(val)[1])
                elseif ku == "CAPACITY"
                    cap = parse(Int, split(val)[1])
                elseif ku == "EDGE_WEIGHT_TYPE"
                    edge_type = strip(split(val)[1])
                end
                continue
            end

            up = uppercase(s)
            if up in ("NODE_COORD_SECTION","DEMAND_SECTION","DEPOT_SECTION","EOF")
                section = up
                continue
            end

            if section == "NODE_COORD_SECTION"
                p = split(s); idx = parse(Int, p[1])
                coords[idx] = (parse(Float64, p[2]), parse(Float64, p[3]))
            elseif section == "DEMAND_SECTION"
                p = split(s); idx = parse(Int, p[1])
                demand[idx] = parse(Int, p[2])
            elseif section == "DEPOT_SECTION"
                idx = parse(Int, split(s)[1])
                if idx != -1; depot = idx; end
            end
        end
    end

    @assert n > 0 "DIMENSION not found in $path"
    @assert cap > 0 "CAPACITY not found in $path"

    coord_vec = [coords[i] for i in 1:n]
    dem_vec   = [get(demand, i, 0) for i in 1:n]

    dist = zeros(Float64, n, n)
    for i in 1:n, j in 1:n
        if i == j
            dist[i,j] = 0.0
        else
            dx = coord_vec[i][1] - coord_vec[j][1]
            dy = coord_vec[i][2] - coord_vec[j][2]
            d  = sqrt(dx*dx + dy*dy)
            et = uppercase(edge_type)
            dist[i,j] = et == "EUC_2D" ? floor(d + 0.5) :
                        et == "CEIL_2D" ? ceil(d) : d
        end
    end

    return VRPInstance(name, n, cap, coord_vec, dem_vec, depot, dist)
end

end # module