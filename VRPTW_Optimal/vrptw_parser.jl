module VRPTWLIB

export VRPTWInstance, parse_solomon, euclidean_distance, n_customers

using LinearAlgebra

struct VRPTWInstance
    name::String
    coords::Matrix{Float64}   # row 1 is depot; rows 2..n+1 are customers
    demand::Vector{Int}
    ready::Vector{Float64}
    due::Vector{Float64}
    service::Vector{Float64}
    capacity::Int
    vehicles::Int
    depot::Int
end

n_customers(inst::VRPTWInstance) = size(inst.coords, 1) - 1

function euclidean_distance(inst::VRPTWInstance, i::Int, j::Int)
    dx = inst.coords[i, 1] - inst.coords[j, 1]
    dy = inst.coords[i, 2] - inst.coords[j, 2]
    return hypot(dx, dy)
end

"""
    parse_solomon(path::AbstractString)

Parse Solomon/Gehring & Homberger style VRPTW benchmark files (e.g., `c101.txt`).
Returns a `VRPTWInstance` with depot at index 1.
"""
function parse_solomon(path::AbstractString)
    lines = [strip(l) for l in readlines(path) if !isempty(strip(l))]
    @assert !isempty(lines) "Empty file: $path"

    name = lines[1]

    # Vehicle section: look for a line with two integers (number, capacity)
    veh_idx = findfirst(l -> occursin("VEHICLE", uppercase(l)), lines)
    @assert veh_idx !== nothing "VEHICLE header not found in $path"
    veh_parts = split(lines[veh_idx + 2])
    @assert length(veh_parts) >= 2 "Malformed vehicle line in $path"
    vehicles = parse(Int, veh_parts[1])
    capacity = parse(Int, veh_parts[end])

    cust_idx = findfirst(l -> startswith(uppercase(l), "CUSTOMER"), lines)
    @assert cust_idx !== nothing "CUSTOMER header not found in $path"
    data_lines = lines[(cust_idx + 2):end]

    rows = [split(replace(l, "," => "")) for l in data_lines]
    rows = [filter(!isempty, r) for r in rows if length(filter(!isempty, r)) >= 7]

    n = length(rows) - 1  # first row is depot
    coords = zeros(Float64, n + 1, 2)
    demand = zeros(Int, n + 1)
    ready = zeros(Float64, n + 1)
    due = zeros(Float64, n + 1)
    service = zeros(Float64, n + 1)

    for r in rows
        id = parse(Int, r[1]) + 1  # shift so depot is 1
        coords[id, 1] = parse(Float64, r[2])
        coords[id, 2] = parse(Float64, r[3])
        demand[id] = parse(Int, r[4])
        ready[id] = parse(Float64, r[5])
        due[id] = parse(Float64, r[6])
        service[id] = parse(Float64, r[7])
    end

    return VRPTWInstance(name, coords, demand, ready, due, service, capacity, vehicles, 1)
end

end # module
