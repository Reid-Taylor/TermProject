import numpy as np
import pandas as pd
import gurobipy as gp
import scipy.stats as ss
import math
import random

supply_uncertainty = np.random.random((50,168,74))
supply_uncertainty = supply_uncertainty * 0.4 + 0.6
demand_uncertainty = np.random.random((50,168,4))
demand_uncertainty = demand_uncertainty * 0.6 + 0.9

arcDistance = pd.read_csv("./arcDistance.csv")
demand = pd.read_csv("./demand.csv")
supply = pd.read_csv("./supplyT.csv")
weighting = pd.read_csv("./weighting.csv")
inverse = pd.read_csv("./inverse_populations.csv")

direct = 1
hub = 0.6 
threshold = 200
H = 78.91
alpha = 0.85
mu = 250*365
n = 168
p = 74

model = gp.Model()
X = model.addMVar((n, n, p), lb=0)
Y = model.addMVar((n, n, p), lb=0)
Z = model.addMVar((n, n, p), lb=0)
delta = model.addMVar((n,n), vtype=gp.GRB.BINARY)
beta = model.addVar(lb=0, vtype=gp.GRB.INTEGER)

objective=0

for i in range(n):
    model.addConstr((sum(delta[i,:]) <= 5))
    for l in range(n):
        model.addConstr((50000000 * delta[i,l] >= sum(Y[i,l,:])))
        model.addConstr((delta[i,l] <= (1/threshold) * sum(Y[i,l,:]) + sum(Z[i,l,:]) + sum(X[i,l,:]))) 
        objective += arcDistance.iloc[i,l] * (direct * sum(X[i,l,:]) + hub * delta[i,l] * sum(Y[i,l,:]) + direct * sum(Z[i,l,:]) * delta[i,l])
        if l < 50:
            for k in range(p):
                model.addConstr((sum(X[:,i,k]) + sum(Y[:,i,k]) <= abs(supply.iloc[i,k] * supply_uncertainty[l,i,k])))
        if l < 50:
            for k in range(4):
                model.addConstr((beta >= 
                   sum(sum(inverse.iloc[i,0] * (demand.iloc[i,k] * demand_uncertainty[l,i,k] - 
                    10000000 * np.dot((sum(X[:,i,k]) + sum(Z[:,i,k])), weighting))) *
                    np.array([1/41.3, 1/107.38, 1/37.9961, 1/330.4]))))
                 
objective = objective * (1-alpha) * H    
objective += alpha * beta * mu
            
model.setObjective(objective, gp.GRB.MINIMIZE)
model.optimize()

Xval = X.X
Yval = Y.X
Zval = Z.X
deltaVal = delta.X
betaVal = beta.X

Xval = np.array(Xval)
Yval = np.array(Yval)
Zval = np.array(Zval)
deltaVal = np.array(deltaVal)
betaVal = np.array(betaVal)

np.save('Xval.npy', Xval)
np.save('Yval.npy', Yval)
np.save('Zval.npy', Zval)
np.save('deltaVal.npy', deltaVal)
np.save('betaVal.npy', betaVal)
