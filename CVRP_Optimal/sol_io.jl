module SolIO

export parse_solution_customers_with_cost, build_routes_with_depot,
       parse_K_from_filename, basename_safe, stated_cost_from_sol

function parse_solution_customers_with_cost(path::AbstractString)
    routes_customers = Vector{Vector{Int}}()
    stated_cost::Union{Nothing,Float64} = nothing
    cost_pattern = r"(?i)\b(cost|distance|obj(?:ective)?)\b[^0-9\-]*(-?\d+(?:\.\d+)?)"
    open(path, "r") do io
        for ln in eachline(io)
            s = strip(ln)
            if startswith(lowercase(s), "route")
                parts = split(s, ":")
                nodes_str = length(parts) > 1 ? strip(parts[2]) : ""
                if !isempty(nodes_str)
                    custs = [parse(Int, t) for t in split(nodes_str) if !isempty(t)]
                    push!(routes_customers, custs)
                end
            else
                m = match(cost_pattern, s)
                if m !== nothing; stated_cost = parse(Float64, m.captures[2]); end
            end
        end
    end
    return routes_customers, stated_cost
end

build_routes_with_depot(routes_customers::Vector{Vector{Int}}, depot::Int) =
    [vcat([depot], filter(x -> x != depot, rc), [depot]) for rc in routes_customers]

parse_K_from_filename(path::AbstractString) = begin
    m = match(r"-k(\d+)\.vrp$"i, path)
    m === nothing ? nothing : parse(Int, m.captures[1])
end

function basename_safe(p::AbstractString)
    s = replace(p, "\\" => "/")
    i = findlast('/', s)
    i === nothing ? s : s[(i+1):end]
end

function stated_cost_from_sol(sol_path::AbstractString)
    cp = r"(?i)\b(cost|distance|obj(?:ective)?)\b[^0-9\-]*(-?\d+(?:\.\d+)?)"
    sc::Union{Nothing,Float64} = nothing
    open(sol_path, "r") do io
        for ln in eachline(io)
            m = match(cp, strip(ln))
            if m !== nothing; sc = parse(Float64, m.captures[2]); end
        end
    end
    return sc
end

end # module