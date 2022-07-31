from pulp import *
import pandas as pd

data = pd.read_csv("data/cluster.csv", header = 0)


# only existing vertiport and total profit columns
clusterName = data['label']
dataTable = data.iloc[:,-3:].values.tolist()

n_Ve = dict([(i, n[0]) for i,n in enumerate(dataTable)])
total_commute = dict([(i, n[1]) for i,n in enumerate(dataTable)])
total_prof = dict([(i, n[2]) for i,n in enumerate(dataTable)])

prob = LpProblem('MaxProfit', LpMinimize)

# number of existing vertiport
n_eVTOL_Vars = LpVariable.dicts('n_eVTOL', clusterName, 0, cat='Integer')
# number of new vertiport
n_Vn_Vars = LpVariable.dicts('n_Vn', clusterName, 0, cat='Integer')

n_round = 5
c_eVTOL = 100000.0
c_Ve = 12000.0
c_Vn = 200000.0
cap_eVTOL = 4.0
cap_port = 8.0
depreciation_1 = 1/365
depreciation_10 = 1/365/10
depreciation_20 = 1/365/20

# Objection function
prob += lpSum([- total_prof[i] + c_eVTOL * n_eVTOL_Vars[i]*depreciation_10 + c_Ve * n_Ve[i]*depreciation_1 + c_Vn * n_Vn_Vars[i] * depreciation_20 for i in clusterName]), 'Total Profit'

# Constraint
prob += lpSum([cap_eVTOL * n_eVTOL_Vars[i] * n_round >= total_commute[i] for i in clusterName ]), 'Enough Capacity'
prob += lpSum([cap_port * (n_Ve[i] + n_Vn_Vars[i])  >= n_eVTOL_Vars[i] for i in clusterName ]), 'Enough Vertiport'

prob.solve()

