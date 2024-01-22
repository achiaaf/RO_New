import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
import rsome as rso
from rsome import ro, cpt_solver as cp, grb_solver as grb, cpx_solver as cpx, eco_solver as eco
import matplotlib.pyplot as plt
from Functions import AGMC_13, MAPE, projection, linear_model, linear_param

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''Importing the data'''
population = pd.read_csv('Population.csv')
gdp = pd.read_csv('GDP.csv')
water_consumption = pd.read_csv('Domestic Water Consumption.csv')
year = np.array(population.iloc[0])

'''Predicting Water Consumption for the different Areas'''
time = 15
t = 2020 + time
areas = 1
year_new = np.hstack((year, np.arange(year[-1] + 1, t + 1)))

# Dictionary = {}
# # Dictionary[0] = pd.DataFrame(year_new, columns=['Year'])
# d1 = pd.DataFrame(year_new, columns=['Year'])
# d2 = pd.DataFrame(year_new, columns=['Year'])
# for i in range(1, len(population.columns)):
#     gdp_new = np.hstack((gdp.iloc[:, 1], projection(gdp.iloc[:, 1], year, t)))
#     pop_new = np.hstack((population.iloc[:, i], projection(population.iloc[:, i], year, t)))
#     predict_water1 = AGMC_13(water_consumption.iloc[:, i], gdp_new, pop_new, 0.0069)
#     p = linear_param(water_consumption.iloc[:, i], gdp.iloc[:, 1], population.iloc[:, i])
#     predict_water2 = linear_model(gdp_new, pop_new, p)
#     # Creating a DataFrame
#     d1.insert(loc=len(d1.columns), column=f'Area {i}', value=predict_water1)
#     d2.insert(loc=len(d2.columns), column=f'Area {i}', value=predict_water2)
# d1.to_excel('Linear Prediction Results.xlsx')
#     # pd.merge(d1,(pd.DataFrame(predict_water1, columns=[f'Linear Regression Forecast for Area {i}'], index=year_new)), on=year_new)
# print(d2)
# print(Dictionary)


# '''There districts are divided into 4 where things are done independently'''
# regions = 4

# x1 = np.array(gdp.iloc[:, 1])
# x2 = np.array(population.iloc[:, 1])
# X = np.array([x1, x2, np.ones(len(x1))]).T
# Y = np.array(water_consumption.iloc[:, 1])
# p = inv(X.T @ X) @ X.T @ Y
# y = X @ p
# r2 = 1-np.linalg.norm(y - X @ p)**2/y.shape[0]/np.var(y)
#
# print(y)
# print(Y)

aquifer = pd.read_csv('Aquifer.csv')
quantity = aquifer['Quantity']
sal = aquifer['Salinity']
cost = aquifer['Cost']

area1 = pd.read_csv('Area1.csv')
crops1 = area1['Crops']
crop_water1 = np.tile(area1['Water requirement'], (time, 1)).T
revenue1 = area1['Revenue'].values
sal_tol1 = np.tile(area1['Optimum Salinity'], (time, 1)).T


t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

water_demand = np.round(pd.read_csv('Water_demand.csv').values, 0)
land_min = 10
land_max = 500
tww_sal = 1.2
desal_sal = 0.5
domestic_sal = 1

'''Defining the Model'''
model = ro.Model()

# Define decision variables
q_Sc_1 = model.dvar((len(crops1), time))  # matrix indicating the amount of water from a source allocated to a crop
q_Wc1 = model.dvar(
    (len(crops1), time))  # vector indicating the amount allocated to the various crops from treated wastewater
q_Dc1 = model.dvar(
    (len(crops1), time))  # vector indicating the amount allocated to the various crops from desalinated water
q_S1 = model.dvar(time)
q_D1 = model.dvar(time)  # the amount of desalinated water allocated for domestic use
land1 = model.dvar((len(crops1), time))  # the amount of land allocated for the various crops

recharge = pd.read_csv('Recharge.csv').values

# Finding the mean of the recharge
r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T


# Objective Function
model.max((revenue1.T @ land1).sum() - ((cost[0] * q_Sc_1).sum() + (cost[0] * q_S1).sum() + 0.7 * (q_Dc1.sum() + q_D1).sum() + 0.4 * q_Wc1.sum()))

'''Defining the constraints'''
qD_vars = [q_D1]
qS_vars = [q_S1]
qW_vars = [q_Wc1]
qDc_vars = [q_Dc1]
qSc_vars = [q_Sc_1]
sal_tol = [sal_tol1]
crop_water = [crop_water1]
land_vars = [land1]

cum_supply1 = [q_Sc_1.sum(axis=0)[0] + q_S1[0] + q_Dc1.sum(axis=0) + q_D1[0]]

cum_supply = [cum_supply1]
recharge_cum = [[r_mean[0, 0]], [r_mean[1, 0]], [r_mean[2, 0]], [r_mean[3, 0]]]
for a in range(0, areas):
    for i in range(1, time):
        cum_supply[a].append(
            cum_supply[a][i - 1] + qSc_vars[a].sum(axis=0)[i] + qS_vars[a][i] + qD_vars[a][i] + qDc_vars[a].sum(axis=0)[
                i])
        recharge_cum[a].append(recharge_cum[a][i - 1] + r_mean[a, i])

for i in range(areas):
    # Domestic Demand Constraints
    model.st(qS_vars[i] + qD_vars[i] >= water_demand[i, 0:time])

    # Crop Demand Constraints
    model.st(qSc_vars[i] + qDc_vars[i] + qW_vars[i] >= crop_water[i] * land_vars[i])

    # Quality Constraint
    model.st(sal[i] * qS_vars[i] + desal_sal * qD_vars[i] <= domestic_sal * (qS_vars[i] + qD_vars[i]))
    model.st(sal[i] * qSc_vars[i] + tww_sal * qW_vars[i] + desal_sal * qDc_vars[i] <= sal_tol[i] * (
            qSc_vars[i] + qW_vars[i] + qDc_vars[i]))

    # Sources Constraint
    model.st(qW_vars[i].sum(axis=0) <= 0.6 * water_demand[i, 0:time])
    model.st(qW_vars[i].sum(axis=0) >= 0.5 * (0.6 * water_demand[i, 0:time]))
    # for n in range(time):
    #     model.st(qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= quantity[i] + recharge_cum[n][i] - cum_supply[i][n])
    model.st(qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i] <= np.tile(quantity[i], time)
             + recharge_cum[i] - cum_supply[i][0] - 0.1 * np.tile(quantity[i], time))
    model.st((qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i]).sum() <= 0.9 * (quantity[i]) + np.sum(recharge[i, :]))

    # Land Constraint
    model.st(land_vars[i].sum(axis=0) >= 1)
    model.st(land_vars[i].sum(axis=1) <= 1000)
    model.st(land_vars[i] >= land_min)
    model.st(land_vars[i] <= land_max)

# Non-negativity
q_vars = [q_Sc_1, q_S1, q_Dc1, q_Wc1]
for v in q_vars:
    model.st(v >= 0)

# Solving the model
model.solve()

# Printing the optimal solution
print(f"Optimal Solution: {model.get()}")
folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Deterministic Results'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

crops = [crops1]
for i in range(areas):
    h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(qSc_vars[i].get())], columns=t_set)
    d1 = pd.DataFrame(data=qSc_vars[i].get(), index=crops[i], columns=t_set)
    h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(qW_vars[i].get())], columns=t_set)
    d2 = pd.DataFrame(data=qW_vars[i].get(), index=crops[i], columns=t_set)
    h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(qDc_vars[i].get())], columns=t_set)
    d3 = pd.DataFrame(data=qDc_vars[i].get(), index=crops[i], columns=t_set)
    h_land = pd.DataFrame(index=['Land Allocated', np.sum(land_vars[i].get())], columns=t_set)
    d4 = pd.DataFrame(data=land_vars[i].get(), index=crops[i], columns=t_set)
    d5 = pd.DataFrame(data=qD_vars[i].get(), index=t_set, columns=['Desalinated Water'])
    d6 = pd.DataFrame(data=qS_vars[i].get(), index=t_set, columns=['Groundwater Water'])
    d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
        np.sum(qSc_vars[i].get()) + np.sum(qDc_vars[i].get()) + np.sum(qS_vars[i].get()) + np.sum(qD_vars[i].get())]})
    (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, 'Domestic Use for Area 1.xlsx'))
    (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
                                                                                                   'Crops Output for Area 1.xlsx'))
