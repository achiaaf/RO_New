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
time = 30
t = 2020 + time
areas = 4
year_new = np.hstack((year, np.arange(year[-1] + 1, t + 1)))

aquifer = pd.read_csv('Aquifer.csv')
quantity = aquifer['Quantity']/6
sal = aquifer['Salinity']
cost = aquifer['Cost']

t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

water_demand = np.round(pd.read_csv('Water_demand.csv').values, 0)
land_min = 10
land_max = 500
tww_sal = 1.2
desal_sal = 0.5
domestic_sal = 1

area1 = pd.read_csv('Area1.csv')
crops1 = area1['Crops']
crop_water1 = np.tile(area1['Water requirement'], (time, 1)).T
revenue1 = area1['Revenue'].values
sal_tol1 = np.tile(area1['Optimum Salinity'], (time, 1)).T


model1 = ro.Model()

q_Sc_1 = model1.dvar((len(crops1), time))  # matrix indicating the amount of water from a source allocated to a crop
q_Wc1 = model1.dvar(
    (len(crops1), time))  # vector indicating the amount allocated to the various crops from treated wastewater
q_Dc1 = model1.dvar(
    (len(crops1), time))  # vector indicating the amount allocated to the various crops from desalinated water
q_S1 = model1.dvar(time)
q_D1 = model1.dvar(time)  # the amount of desalinated water allocated for domestic use
land1 = model1.dvar((len(crops1), time))  # the amount of land allocated for the various crops

recharge = pd.read_csv('Recharge.csv').values

# Finding the mean of the recharge
r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T
recharge_cum = [[r_mean[0, 0]], [r_mean[1, 0]], [r_mean[2, 0]], [r_mean[3, 0]]]
for a in range(0, areas):
    for i in range(1, time):
        recharge_cum[a].append(recharge_cum[a][i - 1] + r_mean[a, i])

# Creating the covariance matrix and just extracting the diagonal
cov_matrix = np.diag(np.diag(np.cov(recharge, rowvar=False)))

# Finding the Cholesky decomposition
delta = np.linalg.cholesky(cov_matrix)
r_uncertain1 = model1.rvar(time)
r_set = (rso.norm(r_uncertain1) <= 0.5)


# Objective Function
model1.max(((revenue1.T @ land1).sum(axis=0) -
          ((cost[0] * q_Sc_1).sum(axis=0) + (cost[0] * q_S1).sum(axis=0) + 0.7 * (q_Dc1.sum() + q_D1) + 0.4 * q_Wc1.sum(axis=0))).sum())

'''Defining the constraints'''
cum_supply1 = [q_Sc_1.sum(axis=0)[0] + q_S1[0] + q_Dc1.sum(axis=0) + q_D1[0]]
for i in range(1, time):
    cum_supply1.append(
            cum_supply1[i - 1] + q_Sc_1.sum(axis=0)[i] + q_S1[i] + q_D1[i] + q_Dc1.sum(axis=0)[
                i])


# Domestic Demand Constraints
model1.st(q_S1 + q_D1 >= water_demand[0, 0:time])

# Crop Demand Constraints
model1.st(q_Sc_1 + q_Dc1 + q_Wc1 >= crop_water1 * land1)

# Quality Constraint
model1.st(sal[0] * q_S1 + desal_sal * q_D1 <= domestic_sal * (q_S1 + q_D1))
model1.st(sal[0] * q_Sc_1 + tww_sal * q_Wc1 + desal_sal * q_Dc1 <= 0.8 * sal_tol1 * (
        q_Sc_1 + q_Wc1 + q_Dc1))

# Sources Constraint
model1.st(q_Wc1.sum(axis=0) <= 0.6 * water_demand[0, 0:time])
model1.st(q_Wc1.sum(axis=0) >= 0.5 * (0.6 * water_demand[0, 0:time]))
# for n in range(time):
#     model.st(qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= quantity[i] + recharge_cum[n][i] - cum_supply[i][n])
model1.st((q_Sc_1.sum(axis=0) + q_Dc1.sum(axis=0) + q_S1 + q_D1 <= np.tile(quantity[0], time)
         + recharge_cum[0] + (delta[0, 0] * r_uncertain1) - cum_supply1[0] - 0.1 * np.tile(quantity[0], time)).forall(r_set))
model1.st((q_Sc_1.sum(axis=0) + q_Dc1.sum(axis=0) + q_S1 + q_D1).sum() <= 0.9 * (quantity[0]) + np.sum(recharge[0, :]))

# Land Constraint
model1.st(land1.sum(axis=0) >= 1)
model1.st(land1.sum(axis=1) <= 1000)
model1.st(land1 >= land_min)
model1.st(land1 <= land_max)

# Non-negativity
q_vars = [q_Sc_1, q_S1, q_Dc1, q_Wc1, q_D1]
for v in q_vars:
    model1.st(v >= 0)

# # Solving the model
# model1.solve(cpx)
#
# # Printing the optimal solution
# print(f"Optimal Solution: {model1.get()}")
# folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Results'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#
# h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(q_Sc_1.get())], columns=t_set)
# d1 = pd.DataFrame(data=q_Sc_1.get(), index=crops1, columns=t_set)
# h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(q_Wc1.get())], columns=t_set)
# d2 = pd.DataFrame(data=q_Wc1.get(), index=crops1, columns=t_set)
# h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(q_Dc1.get())], columns=t_set)
# d3 = pd.DataFrame(data=q_Dc1.get(), index=crops1, columns=t_set)
# h_land = pd.DataFrame(index=['Land Allocated', np.sum(land1.get())], columns=t_set)
# d4 = pd.DataFrame(data=land1.get(), index=crops1, columns=t_set)
# d5 = pd.DataFrame(data=q_D1.get(), index=t_set, columns=['Desalinated Water'])
# d6 = pd.DataFrame(data=q_S1.get(), index=t_set, columns=['Groundwater Water'])
# d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
#     np.sum(q_Sc_1.get()) + np.sum(q_Dc1.get()) + np.sum(q_S1.get()) + np.sum(q_D1.get())]})
# (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, 'Domestic Use for Area 1.xlsx'))
# (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
#                                                                                                'Crops Output for Area 1.xlsx'))
#
# '''AREA 2'''
# area2 = pd.read_csv('Area2.csv')
# crops2 = area2['Crops']
# crop_water2 = np.tile(area2['Water requirement'], (time, 1)).T
# revenue2 = area2['Revenue'].values
# sal_tol2 = np.tile(area2['Optimum Salinity'], (time, 1)).T
#
# model2 = ro.Model()
#
# q_Sc_2 = model2.dvar((len(crops2), time))  # matrix indicating the amount of water from a source allocated to a crop
# q_Wc2 = model2.dvar(
#     (len(crops2), time))  # vector indicating the amount allocated to the various crops from treated wastewater
# q_Dc2 = model2.dvar(
#     (len(crops2), time))  # vector indicating the amount allocated to the various crops from desalinated water
# q_S2 = model2.dvar(time)
# q_D2 = model2.dvar(time)  # the amount of desalinated water allocated for domestic use
# land2 = model2.dvar((len(crops2), time))  # the amount of land allocated for the various crops
#
# r_uncertain2 = model2.rvar(time)
# r_set2 = (rso.norm(r_uncertain2) <= 0.25)
#
# # Objective Function
# model2.max(((revenue2.T @ land2).sum(axis=0) -
#           ((cost[1] * q_Sc_2).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) + 0.7 * (q_Dc2.sum() + q_D2) + 0.4 * q_Wc2.sum(axis=0))).sum())
#
# '''Defining the constraints'''
# cum_supply2 = [q_Sc_2.sum(axis=0)[0] + q_S2[0] + q_Dc2.sum(axis=0) + q_D2[0]]
# for i in range(1, time):
#     cum_supply2.append(
#             cum_supply2[i - 1] + q_Sc_2.sum(axis=0)[i] + q_S2[i] + q_D2[i] + q_Dc2.sum(axis=0)[
#                 i])
#
#
# # Domestic Demand Constraints
# model2.st(q_S2 + q_D2 >= water_demand[1, 0:time])
#
# # Crop Demand Constraints
# model2.st(q_Sc_2 + q_Dc2 + q_Wc2 >= crop_water2 * land2)
#
# # Quality Constraint
# model2.st(sal[1] * q_S2 + desal_sal * q_D2 <= domestic_sal * (q_S2 + q_D2))
# model2.st(sal[1] * q_Sc_2 + tww_sal * q_Wc2 + desal_sal * q_Dc2 <= sal_tol2 * (
#         q_Sc_2 + q_Wc2 + q_Dc2))
#
# # Sources Constraint
# model2.st(q_Wc2.sum(axis=0) <= 0.6 * water_demand[1, 0:time])
# model2.st(q_Wc2.sum(axis=0) >= 0.5 * (0.6 * water_demand[1, 0:time]))
#
# model2.st((q_Sc_2.sum(axis=0) + q_Dc2.sum(axis=0) + q_S2 + q_D2 <= np.tile(quantity[1], time)
#          + recharge_cum[1] + (delta[1, 1] * r_uncertain2) - cum_supply2[0] - 0.1 * np.tile(quantity[1], time)).forall(r_set2))
# model2.st((q_Sc_2.sum(axis=0) + q_Dc2.sum(axis=0) + q_S2 + q_D2).sum() <= 0.9 * (quantity[1]) + np.sum(recharge[1, :]))
#
# # Land Constraint
# model2.st(land2.sum(axis=0) >= 1)
# model2.st(land2.sum(axis=1) <= 1000)
# model2.st(land2 >= land_min)
# model2.st(land2 <= land_max)
#
# # Non-negativity
# q_vars = [q_Sc_2, q_S2, q_Dc2, q_Wc2, q_D2]
# for v in q_vars:
#     model2.st(v >= 0)
#
# # Solving the model
# model2.solve(cpx)
#
# # Printing the optimal solution
# print(f"Optimal Solution: {model2.get()}")
# folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Results'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#
# h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(q_Sc_2.get())], columns=t_set)
# d1 = pd.DataFrame(data=q_Sc_2.get(), index=crops2, columns=t_set)
# h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(q_Wc2.get())], columns=t_set)
# d2 = pd.DataFrame(data=q_Wc2.get(), index=crops2, columns=t_set)
# h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(q_Dc2.get())], columns=t_set)
# d3 = pd.DataFrame(data=q_Dc2.get(), index=crops2, columns=t_set)
# h_land = pd.DataFrame(index=['Land Allocated', np.sum(land2.get())], columns=t_set)
# d4 = pd.DataFrame(data=land2.get(), index=crops2, columns=t_set)
# d5 = pd.DataFrame(data=q_D2.get(), index=t_set, columns=['Desalinated Water'])
# d6 = pd.DataFrame(data=q_S2.get(), index=t_set, columns=['Groundwater Water'])
# d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
#     np.sum(q_Sc_2.get()) + np.sum(q_Dc2.get()) + np.sum(q_S2.get()) + np.sum(q_D2.get())]})
# (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, 'Domestic Use for Area 2.xlsx'))
# (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
#                                                                                                'Crops Output for Area 2.xlsx'))
#
#
# '''AREA 3'''
# area3 = pd.read_csv('Area3.csv')
# crops3 = area3['Crops']
# crop_water3 = np.tile(area3['Water requirement'], (time, 1)).T
# revenue3 = area3['Revenue'].values
# sal_tol3 = np.tile(area3['Optimum Salinity'], (time, 1)).T
#
# model3 = ro.Model()
#
# q_Sc_3 = model3.dvar((len(crops3), time))  # matrix indicating the amount of water from a source allocated to a crop
# q_Wc3 = model3.dvar(
#     (len(crops3), time))  # vector indicating the amount allocated to the various crops from treated wastewater
# q_Dc3 = model3.dvar(
#     (len(crops3), time))  # vector indicating the amount allocated to the various crops from desalinated water
# q_S3 = model3.dvar(time)
# q_D3 = model3.dvar(time)  # the amount of desalinated water allocated for domestic use
# land3 = model3.dvar((len(crops3), time))  # the amount of land allocated for the various crops
#
# r_uncertain3 = model3.rvar(time)
# r_set3 = (rso.norm(r_uncertain3) <= 0.25)
#
# # Objective Function
# model3.max(((revenue3.T @ land3).sum(axis=0) -
#           ((cost[3] * q_Sc_3).sum(axis=0) + (cost[3] * q_S3).sum(axis=0) + 0.7 * (q_Dc3.sum() + q_D3) + 0.4 * q_Wc3.sum(axis=0))).sum())
#
# '''Defining the constraints'''
# cum_supply3 = [q_Sc_3.sum(axis=0)[0] + q_S3[0] + q_Dc3.sum(axis=0) + q_D3[0]]
# for i in range(1, time):
#     cum_supply3.append(
#             cum_supply3[i - 1] + q_Sc_3.sum(axis=0)[i] + q_S3[i] + q_D3[i] + q_Dc3.sum(axis=0)[
#                 i])
#
#
# # Domestic Demand Constraints
# model3.st(q_S3 + q_D3 >= water_demand[2, 0:time])
#
# # Crop Demand Constraints
# model3.st(q_Sc_3 + q_Dc3 + q_Wc3 >= crop_water3 * land3)
#
# # Quality Constraint
# model3.st(sal[1] * q_S3 + desal_sal * q_D3 <= domestic_sal * (q_S3 + q_D3))
# model3.st(sal[1] * q_Sc_3 + tww_sal * q_Wc3 + desal_sal * q_Dc3 <= sal_tol3 * (
#         q_Sc_3 + q_Wc3 + q_Dc3))
#
# # Sources Constraint
# model3.st(q_Wc3.sum(axis=0) <= 0.6 * water_demand[2, 0:time])
# model3.st(q_Wc3.sum(axis=0) >= 0.5 * (0.6 * water_demand[2, 0:time]))
#
# model3.st((q_Sc_3.sum(axis=0) + q_Dc3.sum(axis=0) + q_S3 + q_D3 <= np.tile(quantity[2], time)
#          + recharge_cum[1] + (delta[1, 1] * r_uncertain3) - cum_supply3[0] - 0.1 * np.tile(quantity[2], time)).forall(r_set3))
# model3.st((q_Sc_3.sum(axis=0) + q_Dc3.sum(axis=0) + q_S3 + q_D3).sum() <= 0.9 * (quantity[2]) + np.sum(recharge[2, :]))
#
# # Land Constraint
# model3.st(land3.sum(axis=0) >= 1)
# model3.st(land3.sum(axis=1) <= 1000)
# model3.st(land3 >= land_min)
# model3.st(land3 <= land_max)
#
# # Non-negativity
# q_vars = [q_Sc_3, q_S3, q_Dc3, q_Wc3, q_D3]
# for v in q_vars:
#     model3.st(v >= 0)
#
# # Solving the model
# model3.solve(cpx)
#
# # Printing the optimal solution
# print(f"Optimal Solution: {model3.get()}")
# folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Results'
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)
#
# h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(q_Sc_3.get())], columns=t_set)
# d1 = pd.DataFrame(data=q_Sc_3.get(), index=crops3, columns=t_set)
# h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(q_Wc3.get())], columns=t_set)
# d2 = pd.DataFrame(data=q_Wc3.get(), index=crops3, columns=t_set)
# h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(q_Dc3.get())], columns=t_set)
# d3 = pd.DataFrame(data=q_Dc3.get(), index=crops3, columns=t_set)
# h_land = pd.DataFrame(index=['Land Allocated', np.sum(land3.get())], columns=t_set)
# d4 = pd.DataFrame(data=land3.get(), index=crops3, columns=t_set)
# d5 = pd.DataFrame(data=q_D3.get(), index=t_set, columns=['Desalinated Water'])
# d6 = pd.DataFrame(data=q_S3.get(), index=t_set, columns=['Groundwater Water'])
# d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
#     np.sum(q_Sc_3.get()) + np.sum(q_Dc3.get()) + np.sum(q_S3.get()) + np.sum(q_D3.get())]})
# (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, 'Domestic Use for Area 3.xlsx'))
# (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
#                                                                                                'Crops Output for Area 3.xlsx'))


'''AREA 4'''
area4 = pd.read_csv('Area4.csv')
crops4 = area4['Crops']
crop_water4 = np.tile(area4['Water requirement'], (time, 1)).T
revenue4 = area4['Revenue'].values
sal_tol4 = np.tile(area4['Optimum Salinity'], (time, 1)).T

model4 = ro.Model()

q_Sc_4 = model4.dvar((len(crops4), time))  # matrix indicating the amount of water from a source allocated to a crop
q_Wc4 = model4.dvar(
    (len(crops4), time))  # vector indicating the amount allocated to the various crops from treated wastewater
q_Dc4 = model4.dvar(
    (len(crops4), time))  # vector indicating the amount allocated to the various crops from desalinated water
q_S4 = model4.dvar(time)
q_D4 = model4.dvar(time)  # the amount of desalinated water allocated for domestic use
land4 = model4.dvar((len(crops4), time))  # the amount of land allocated for the various crops

r_uncertain4 = model4.rvar(time)
r_set4 = (rso.norm(r_uncertain4) <= 0.25)

# Objective Function
model4.max(((revenue4.T @ land4).sum(axis=0) -
          ((cost[3] * q_Sc_4).sum(axis=0) + (cost[3] * q_S4).sum(axis=0) + 0.7 * (q_Dc4.sum() + q_D4) + 0.4 * q_Wc4.sum(axis=0))).sum())

'''Defining the constraints'''
cum_supply4 = [q_Sc_4.sum(axis=0)[0] + q_S4[0] + q_Dc4.sum(axis=0) + q_D4[0]]
for i in range(1, time):
    cum_supply4.append(
            cum_supply4[i - 1] + q_Sc_4.sum(axis=0)[i] + q_S4[i] + q_D4[i] + q_Dc4.sum(axis=0)[
                i])


# Domestic Demand Constraints
model4.st(q_S4 + q_D4 >= water_demand[3, 0:time])

# Crop Demand Constraints
model4.st(q_Sc_4 + q_Dc4 + q_Wc4 >= crop_water4 * land4)

# Quality Constraint
model4.st(sal[3] * q_S4 + desal_sal * q_D4 <= domestic_sal * (q_S4 + q_D4))
model4.st(sal[3] * q_Sc_4 + tww_sal * q_Wc4 + desal_sal * q_Dc4 <= sal_tol4 * (
        q_Sc_4 + q_Wc4 + q_Dc4))

# Sources Constraint
model4.st(q_Wc4.sum(axis=0) <= 0.6 * water_demand[3, 0:time])
# model4.st(q_Wc4.sum(axis=0) >= 0.5 * (0.6 * water_demand[3, 0:time]))

model4.st((q_Sc_4.sum(axis=0) + q_Dc4.sum(axis=0) + q_S4 + q_D4 <= np.tile(quantity[3], time)
         + recharge_cum[1] + (delta[1, 1] * r_uncertain4) - cum_supply4[0] - 0.1 * np.tile(quantity[3], time)).forall(r_set4))


# Land Constraint
model4.st(land4.sum(axis=0) >= 1)
model4.st(land4.sum(axis=1) <= 1000)
model4.st(land4 >= land_min)
model4.st(land4 <= land_max)


# Non-negativity
q_vars = [q_Sc_4, q_S4, q_Dc4, q_Wc4, q_D4]
for v in q_vars:
    model4.st(v >= 0)

# Solving the model
model4.solve(grb)

print((sal[3] * q_Sc_4.get() + tww_sal*q_Wc4.get() + desal_sal * q_Dc4.get())/(q_Sc_4.get()+q_Wc4.get()+q_Dc4.get()))
print(sal_tol4)
# Printing the optimal solution
print(f"Optimal Solution: {model4.get()}")
folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Results'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(q_Sc_4.get())], columns=t_set)
d1 = pd.DataFrame(data=q_Sc_4.get(), index=crops4, columns=t_set)
h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(q_Wc4.get())], columns=t_set)
d2 = pd.DataFrame(data=q_Wc4.get(), index=crops4, columns=t_set)
h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(q_Dc4.get())], columns=t_set)
d3 = pd.DataFrame(data=q_Dc4.get(), index=crops4, columns=t_set)
h_land = pd.DataFrame(index=['Land Allocated', np.sum(land4.get())], columns=t_set)
d4 = pd.DataFrame(data=land4.get(), index=crops4, columns=t_set)
d5 = pd.DataFrame(data=q_D4.get(), index=t_set, columns=['Desalinated Water'])
d6 = pd.DataFrame(data=q_S4.get(), index=t_set, columns=['Groundwater Water'])
d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
    np.sum(q_Sc_4.get()) + np.sum(q_Dc4.get()) + np.sum(q_S4.get()) + np.sum(q_D4.get())]})
(pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, 'Domestic Use for Area 4grb1.xlsx'))
(pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
                                                                                               'Crops Output for Area 4grb1.xlsx'))






# recharge = pd.read_csv('Recharge.csv').values
#
# # Finding the mean of the recharge
# r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T
#
# # Creating the covariance matrix and just extracting the diagonal
# cov_matrix = np.diag(np.diag(np.cov(recharge, rowvar=False)))
#
# revenue = [revenue1, revenue2, revenue3, revenue4]
# sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
# crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
# crops = [crops1, crops2, crops3, crops4]
#
#
#
# for i in range(areas):
#     '''Defining the Model'''
#     model = ro.Model()
#
#     # Define decision variables
#     q_Sc = model.dvar((len(crops[i]), time))  # matrix indicating the amount of water from a source allocated to a crop
#     q_Wc = model.dvar(
#         (len(crops[i]), time))  # vector indicating the amount allocated to the various crops from treated wastewater
#     q_Dc = model.dvar(
#         (len(crops[i]), time))  # vector indicating the amount allocated to the various crops from desalinated water
#     q_S = model.dvar(time)  # vector indicating the amount of water allocated for domestic use for the various sources
#     q_D = model.dvar(time)  # the amount of desalinated water allocated for domestic use
#     land = model.dvar((len(crops[i]), time))  # the amount of land allocated for the various crops
#
#     # Finding the Cholesky decomposition
#     delta = np.linalg.cholesky(cov_matrix)
#     r_uncertain = model.rvar(time)
#     r_set = (rso.norm(r_uncertain) <= 0.25)
#
#     # Objective Function
#     model.max(((revenue[i].T @ land).sum()).sum() - ((cost[i] * q_Sc).sum() + (cost[i] * q_S).sum() + 0.7 * (q_Dc.sum() + q_D) + 0.4 * q_Wc.sum()).sum())
#
#     '''Defining the constraints'''
#     cum_supply = [q_Sc.sum(axis=0)[0] + q_S[0] + q_Dc.sum(axis=0) + q_D[0]]
#     recharge_cum = [r_mean[i, 0]]
#     for n in range(1, time):
#         cum_supply.append(
#             cum_supply[n - 1] + q_Sc.sum(axis=0)[n] + q_S[n] + q_D[n] + q_Dc.sum(axis=0)[
#                 n])
#         recharge_cum.append(recharge_cum[n - 1] + r_mean[i, n])
#
#     # Domestic Demand Constraints
#     model.st(q_S + q_D >= water_demand[i, 0:time])
#
#     # Crop Demand Constraints
#     model.st(q_Sc + q_Dc + q_Wc >= crop_water[i] * land)
#
#     # Quality Constraint
#     model.st(sal[i] * q_S + desal_sal * q_D <= domestic_sal * (q_S + q_D))
#     model.st(sal[i] * q_Sc + tww_sal * q_Wc + desal_sal * q_Dc <= sal_tol[i] * (
#             q_Sc + q_Wc + q_Dc))
#
#     # Sources Constraint
#     model.st(q_Wc.sum(axis=0) <= 0.6 * water_demand[i, 0:time])
#     model.st(q_Wc.sum(axis=0) >= 0.5 * (0.6 * water_demand[i, 0:time]))
#     # for n in range(time):
#     #     model.st(qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= quantity[i] + recharge_cum[n][i] - cum_supply[i][n])
#     model.st((q_Sc.sum(axis=0) + q_Dc.sum(axis=0) + q_S + q_D <= np.tile(quantity[i], time)
#              + recharge_cum[i] + (delta[i, i] * r_uncertain[i]) - cum_supply[i][0] - 0.1 * np.tile(quantity[i], time)).forall(r_set))
#     model.st((q_Sc.sum(axis=0) + q_Dc.sum(axis=0) + q_S + q_D).sum() <= 0.9 * (quantity[i]) + np.sum(recharge[i, :]))
#
#     # Land Constraint
#     model.st(land.sum(axis=0) >= 1)
#     model.st(land.sum(axis=1) <= 1000)
#     model.st(land >= land_min)
#     model.st(land <= land_max)
#
#     # Non-negativity
#     q_vars = [q_Sc, q_S, q_Dc, q_D, q_Wc]
#     for v in q_vars:
#         model.st(v >= 0)
#
#     # Solving the model
#     model.solve(cpx)
#
#     # Printing the optimal solution
#     print(f"Optimal Solution: {model.get()}")
#     folder_path = f'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Results{i+1}'
#     if not os.path.exists(folder_path):
#         os.makedirs(folder_path)
#
#     h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(q_Sc.get())], columns=t_set)
#     d1 = pd.DataFrame(data=q_Sc.get(), index=crops[i], columns=t_set)
#     h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(q_Wc.get())], columns=t_set)
#     d2 = pd.DataFrame(data=q_Wc.get(), index=crops[i], columns=t_set)
#     h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(q_Dc.get())], columns=t_set)
#     d3 = pd.DataFrame(data=q_Dc.get(), index=crops[i], columns=t_set)
#     h_land = pd.DataFrame(index=['Land Allocated', np.sum(land.get())], columns=t_set)
#     d4 = pd.DataFrame(data=land.get(), index=crops[i], columns=t_set)
#     d5 = pd.DataFrame(data=q_D.get(), index=t_set, columns=['Desalinated Water'])
#     d6 = pd.DataFrame(data=q_S.get(), index=t_set, columns=['Groundwater Water'])
#     d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
#         np.sum(q_Sc.get()) + np.sum(q_Dc.get()) + np.sum(q_S.get()) + np.sum(q_D.get())]})
#     (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, f'Domestic Use for Area {i + 1}.xlsx'))
#     (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
#                                                                                                    f'Crops Output for Area {i + 1}.xlsx'))
