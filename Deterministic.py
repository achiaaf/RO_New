import pandas as pd
import numpy as np
from rsome import ro
from Functions import projection, AGMC_13

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Create a minimization problem
prob = ro.Model()
data = pd.read_csv('LP.csv')
cost = data['Cost ($/m3)'][:4].values.T
capacity = data['Flow rate (m3/year)'][:4].values * 0.03
source_salinity = data['Salinity (dS/m)'][:4].values.T
crop_salinity = data['Optimum Salinity Tolerance (dS/m)'][:7]
crop_water_requirement = data['Irrigation Water Requirements (m3/hectare/year)'][:7].values
sources = data['Source'][:4]
crops = data['Crop'][:7]
prices = data['Revenue ($/hectare)'][:7].values
expenses = data['Expenses ($/hectare)'][:7].values


# Importing and defining the data for domestic demand prediction
data1 = pd.read_csv('Arava Data.csv')
Time = data1['Year'].values
water_consumption = data1['Water Consumption'].values
water_consumption_train = np.array(water_consumption)[:8]
gdp = data1['GDP'].values
population = data1['Population'].values
duration = 2020

# Predicting the water consumption
Y_new = np.hstack((gdp, projection(gdp, Time, duration)))
Z_new = np.hstack((population, projection(population, Time, duration)))
Time_new = np.hstack((Time, np.arange(Time[-1] + 1, duration + 1, 1)))
Predicted_water_consumption = AGMC_13(water_consumption_train, Y_new, Z_new, 0.001)
demand_mean = np.mean(Predicted_water_consumption) * 1000 * 1.5

# Importing and defining the recharge data
data = pd.read_csv('Aquifer Recharge.csv')
columns = [' En Yahav', 'Paran', 'Hazeva', ' Idan']
data_values = data[columns].values

# Finding the mean of the recharge
mean_vector = np.mean(data_values, axis=0)

# Define decision variables
q = prob.dvar((len(sources), len(crops)), vtype='I')  # matrix indicating the amount of water from a source allocated to a crop
d = prob.dvar(len(sources), vtype='I')  # matrix indicating the amount of water allocated from each source for domestic use
d_c = prob.dvar(len(crops), vtype='I')  # matrix indicating the amount of desalinated water allocated for the crops
d_d = prob.dvar(vtype='I')  # amount of desalinated water allocated for domestic use
t_w = prob.dvar(len(crops), vtype='I')  # matrix indicating the amount of treated wastewater allocated for each crop
land = prob.dvar(len(crops), vtype='I')


# Define the objective function with a stochastic cost variable
# prob.min(((cost + np.random.normal(loc=0, scale=0.1, size=cost.shape)) @ q).sum() + ((cost + np.random.normal(loc=0, scale=0.1, size=cost.shape)) * d).sum())

# Define the objective function
prob.max((prices - expenses) @ land - ((cost @ q).sum() + (cost @ d).sum() + (0.45 * t_w).sum() + (0.68 * d_c).sum() + 0.68 * d_d))

# Define the constraints
prob.st(q.sum(axis=0) + t_w + d_c >= crop_water_requirement * land)  # The irrigation water requirements
prob.st(d.sum() + d_d >= demand_mean)  # demand constraint
prob.st(q.sum(axis=1) + d <= np.array(capacity) + mean_vector)  # the water sources constraint
prob.st(q.sum(axis=1) + d >= 0.1 * capacity)
prob.st(q.sum() + d.sum() + d_c.sum() + d_d <= np.sum(capacity + mean_vector))  #
prob.st(t_w.sum() <= 0.6 * demand_mean)
prob.st(t_w <= 0.5 * crop_water_requirement * land)

# Quality constraint
for i in range(len(crops)):
    prob.st(source_salinity @ q[:, i] + 0.7 * t_w[i] + 0.25 * d_c[i] <= crop_salinity[i] * (q[:, i].sum() + t_w[i] + d_c[i]))
prob.st(source_salinity @ d + 0.25 * d_d <= 0.5 * (d.sum() + d_d))


prob.st(q >= 0)  # non-negativity constraint
prob.st(d >= 0)
prob.st(d_d >= 0)
prob.st(d_c >= 0)
prob.st(t_w >= 0)
prob.st(land >= 2)
prob.st(land <= 100)
prob.st(land.sum() <= 500)
# prob.st(0.1 * min_quantity <= max_yield * land <= 0.6 * min_quantity)
# prob.st(land.sum() <= 10000)


# Solve the problem
prob.solve()

# # Print the optimal solution
print(f"Optimal Solution: {prob.get()}")
# print(q.get())
# print(d.get())
# print(land.get())

# Creating a dataframe for the results and exporting to excel
d1 = pd.DataFrame(data=q.get(), index=sources, columns=crops)
d2 = pd.DataFrame(data=t_w.get(), index=crops, columns=['TWW'])
d3 = pd.DataFrame(data=d_c.get(), index=crops, columns=['Desalinated Water'])
cw = pd.DataFrame(data=crop_water_requirement*land.get(), index=crops, columns=['Crop Water Requirement'])
d4 = pd.DataFrame(data=land.get(), index=crops, columns=['Land Allocated'])
d5 = pd.DataFrame(data=d.get(), index=sources, columns=['Domestic Use'])
d51 = pd.DataFrame({'Domestic Use': [0, d_d.get()]}, index=['TWW', 'Desalinated Water'])
df = (pd.concat([d2, d3, cw], axis=1)).transpose()
d_c = pd.concat([d1, df], axis=0)
d52 = pd.concat([d5, d51])
dl = pd.DataFrame({'Domestic Use': [0]}, index=['Land Allocated'])
dl2 = pd.concat([d4.transpose(), dl], axis=1)
d_c1 = pd.concat([d_c, d52], axis=1)
d_c2 = pd.concat([d_c1, dl2], axis=0)
d_c2.to_excel('Solution_wo_Uncertainty_Demand_increase.xlsx')
