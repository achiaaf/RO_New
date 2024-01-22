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

aquifer = pd.read_csv('Aquifer.csv')
quantity = aquifer['Quantity']/4
sal = aquifer['Salinity']
cost = aquifer['Cost']

area4 = pd.read_csv('Area4.csv')
crops4 = area4['Crops']
crop_water4 = np.tile(area4['Water requirement'], (time, 1)).T
revenue4 = area4['Revenue'].values
sal_tol4 = np.tile(area4['Optimum Salinity'], (time, 1)).T
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
q_Sc_4 = model.dvar((len(crops4), time))
q_Wc4 = model.dvar(
    (len(crops4), time))  # vector indicating the amount allocated to the various crops from treated wastewater
q_Dc4 = model.dvar(
    (len(crops4), time))  # vector indicating the amount allocated to the various crops from desalinated water
q_S4 = model.dvar(time)  # vector indicating the amount of water allocated for domestic use for the various sources
q_D4 = model.dvar(time)  # the amount of desalinated water allocated for domestic use
land4 = model.dvar((len(crops4), time))  # the amount of land allocated for the various crops

recharge = pd.read_csv('Recharge.csv').values

# Finding the mean of the recharge
r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T

# Creating the covariance matrix and just extracting the diagonal
cov_matrix = np.diag(np.diag(np.cov(recharge, rowvar=False)))

# Finding the Cholesky decomposition
delta = np.linalg.cholesky(cov_matrix)
r_uncertain4 = model.rvar(time)
r_set = (rso.norm(r_uncertain4) <= 1.2)


# Objective Function
model.max((((revenue4.T @ land4).sum()).sum() -
          ((cost[3] * q_Sc_4).sum() + (cost[3] * q_S4).sum()
           + 0.7 * (q_Dc4.sum() + q_D4) + 0.4 * q_Wc4.sum()).sum()))

'''Defining the constraints'''
qD_vars = [q_D4]
qS_vars = [q_S4]
qW_vars = [q_Wc4]
qDc_vars = [q_Dc4]
qSc_vars = [q_Sc_4]
sal_tol = [sal_tol4]
crop_water = [crop_water4]
land_vars = [land4]

cum_supply = [q_Sc_4.sum(axis=0)[0] + q_S4[0] + q_Dc4.sum(axis=0) + q_D4[0]]

recharge_cum = [[r_mean[0, 0]], [r_mean[1, 0]], [r_mean[2, 0]], [r_mean[3, 0]]]
for a in range(0, areas):
    for i in range(1, time):
        cum_supply.append(
            cum_supply[i - 1] + qSc_vars[a].sum(axis=0)[i] + qS_vars[a][i] + qD_vars[a][i] + qDc_vars[a].sum(axis=0)[
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
    model.st((qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i] <= np.tile(quantity[i], time)
             + recharge_cum[i] + (delta[i, i] * r_uncertain4) - cum_supply[i] - 0.1 * np.tile(quantity[i], time)).forall(r_set))
    # model.st((qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i]).sum() <= 0.9 * (quantity[i]) + np.sum(recharge[i, :]))

    # Land Constraint
    model.st(land_vars[i].sum(axis=0) >= 1)
    model.st(land_vars[i].sum(axis=1) <= 1000)
    model.st(land_vars[i] >= land_min)
    model.st(land_vars[i] <= land_max)

# Non-negativity
q_vars = [q_Sc_4, q_S4, q_Dc4, q_D4, q_Wc4]
for v in q_vars:
    model.st(v >= 0)

# Solving the model
model.solve(grb)
r = 1
# Printing the optimal solution
print(f"Optimal Solution: {model.get()}")
folder_path = f'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Area 4 Results{r}'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

crops = [crops4]
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
    (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, f'Domestic Use {i + 1}.xlsx'))
    (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
                                                                                                   f'Crops Output {i + 1}.xlsx'))
