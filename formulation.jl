using Gurobi, JuMP, DataFrames, CSV, Plots
function distance(x1, y1, x2, y2)
    floatpointcorrection = sin(y1)*sin(y2) + cos(y1)*cos(y2)*cos(x1-x2)
    floatpointcorrection = ifelse(floatpointcorrection > 1, 1.0, floatpointcorrection)
    floatpointcorrection = ifelse(floatpointcorrection < 0, 0.0, floatpointcorrection)
    ß = acos(floatpointcorrection)
    return ß * 6378
end;
function distance(x1, y1, x2, y2)
    floatpointcorrection = sin(y1)*sin(y2) + cos(y1)*cos(y2)*cos(x1-x2)
    floatpointcorrection = ifelse(floatpointcorrection > 1, 1.0, floatpointcorrection)
    floatpointcorrection = ifelse(floatpointcorrection < 0, 0.0, floatpointcorrection)
    ß = acos(floatpointcorrection)
    return ß * 6378
end;
demand = combine(groupby(data[!,1:6], :Area), first)
for row in eachrow(demand)
    row[:fats] = 365 * row[:fats]
    row[:proteins] = 365 * row[:proteins]
    row[:carbs] = 365 * row[:carbs]
    row[:fruits_and_veggies] = 365 * row[:fruits_and_veggies]
end
demand = Matrix(demand[:,2:5]);
loc = unique(data[!,[6,8,9]]);
supply = data[:,[6,7,10]]
supplyT = DataFrame(zeros((length(unique(supply[:,1])), length(unique(supply[:,2])))), :auto)
rename!(supplyT, unique(supply[:,2]));
supplyg = groupby(supply, :Area)
for i in 1:168
    for row in eachrow(supplyg[i])
        supplyT[i, Symbol(row[2])] = row[3]
    end
end
inverse_populations = inv.(unique(population)[!,2])
weighting = CSV.read("./data/weighting.csv", DataFrame)
weighting = Matrix(weighting[:,2:end])
n,p1 = size(supplyT)
n,p2 = size(demand)
lambda = 250*365;
arcDistance = DataFrame(zeros(n, n), :auto)
for i in 1:168
    for j in 1:168
        arcDistance[i, j] = distance(loc[i,2], loc[i,3], loc[j,2], loc[j,3])
    end
end
arcDistance = Matrix(round.(arcDistance, digits=7));
øD = 1 # relative scaling of price per unit when transporting to directly
øH = 0.8 # relative scaling of price per unit when transporting to a central hub as a point of distribution
threshold = 3970 # capacity of 10 Boeing 747-400Fs
H = 78.91 # price of per km per metric tonne transport via a Boeing 747-400F
å = 0.85 # relative importance between the two objectives
using Random
supply_uncertainty = randn!(MersenneTwister(1234), zeros(50, 168, 74))
supply_uncertainty = round.(abs.(supply_uncertainty .* inv.(maximum(abs.(supply_uncertainty)))) .* 0.6 .+ 0.9, digits=4)
demand_uncertainty = randn!(MersenneTwister(1234), zeros(50, 168, 4))
demand_uncertainty = round.(abs.(demand_uncertainty .* inv.(maximum(abs.(demand_uncertainty)))) .* 0.6 .+ 0.9, digits=4);
uncertainty=50
model = Model(Gurobi.Optimizer )
@variable(model, X[i=1:n,j=1:n,k=1:p1] .≥ 0)
@variable(model, Y[i=1:n,j=1:n,k=1:p1] .≥ 0)
@variable(model, Z[i=1:n,j=1:n,k=1:p1] .≥ 0)
@variable(model, ∂[i=1:n,j=1:n] .≥ 0, binary=true)
@variable(model, ß ≥ 0, integer=true)
@constraint(model, auxiliary[z=1:uncertainty], ß ≥ inverse_populations' * (demand .* demand_uncertainty[z,:,:] - (sum(X[i,:,:] for i=1:n) + sum(Z[l,:,:] for l=1:n)) * 10000000 * weighting) * [1/41.3, 1/107.38, 1/37.9961, 1/330.4])
@constraint(model, realSupply[i=1:n,k=1:p1,z=1:uncertainty], sum(X[i,j,k] for j=1:n) + sum(Y[i,l,k] for l=1:n) ≤ abs.(supplyT .* supply_uncertainty[z,:,:])[i,k])
@constraint(model, realDemand[z=1:uncertainty], (sum(X[i,:,:] for i=1:n) + sum(Z[i,:,:] for i=1:n)) * 10000000 * weighting .≤ demand .* demand_uncertainty[z,:,:])
@constraint(model, hubIndicator, ∂ .≤ inv(threshold) .* sum(Y[:,:,k] + Z[:,:,k] + X[:,:,k] for k=1:p1))
@objective(model, Min, (1-å) * H * 
   (sum(sum(sum(X[i,j,k] for k=1:p1) * arcDistance[i,j] for j=1:n) * øD for i=1:n) + 
    sum(sum(sum(Y[i,l,k] for k=1:p1) * ∂[i,l] * arcDistance[i,l] for l=1:n) * øH for i=1:n) + 
    sum(sum(sum(Z[l,j,k] for k=1:p1) * ∂[l,j] * arcDistance[l,j] for j=1:n) * øD for l=1:n)) + (å) * ß * lambda)
optimize!(model)
