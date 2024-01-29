import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
import rsome as rso
from rsome import ro, grb_solver as grb
from Functions import MAPE, projection, linear_model, linear_param
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Creating a folder to save results
folder_path = f'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/' \
              f'Scripts/RO_New/DemRecharge Uncertainty Resultsn'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


'''Importing the data'''
population = pd.read_csv('Population.csv')
gdp = pd.read_csv('GDP.csv')
water_consumption = pd.read_csv('Domestic Water Consumption.csv')
year = np.array(population.iloc[0])

'''Predicting Water Consumption for the different Areas'''
time = 20
t = 2020 + time
areas = 4
year_new = np.hstack((year, np.arange(year[-1] + 1, t + 1)))
a = 0
year = np.array(population.iloc[a:, 0])

water_demand = []
error = []
for i in range(1, areas+1):
    gdp_new = np.hstack((gdp.iloc[a:, 1], projection(gdp.iloc[a:, 1], year, t)))
    pop_new = np.hstack((population.iloc[a:, i], projection(population.iloc[a:, i], year, t)))
    p1 = linear_param(water_consumption.iloc[a:7, i], gdp.iloc[a:7, 1], population.iloc[a:7, i])
    predict_water = np.round(linear_model(gdp_new, pop_new, p1), 0)
    water_demand.append(predict_water[8:])
    mape = MAPE(water_consumption.iloc[:, i], predict_water[0:len(water_consumption.iloc[:, i])])
    error.append((mape/100))

aquifer = pd.read_csv('Aquifer.csv')
aquifer_prod = pd.read_csv('Yearly Production.csv')
quantity = aquifer_prod.mean(axis=0)
aquifer_sal = pd.read_csv('Yearly Salinity.csv')
sal = aquifer_sal.mean(axis=0)
cost = aquifer['Cost']

area1 = pd.read_csv('Area1.csv')
crops1 = area1['Crops']
crop_water1 = np.tile(area1['Water requirement'], (time, 1)).T
revenue1 = area1['Revenue'].values
sal_tol1 = np.tile(area1['Optimum Salinity'], (time, 1)).T

area2 = pd.read_csv('Area2.csv')
crops2 = area2['Crops']
crop_water2 = np.tile(area2['Water requirement'], (time, 1)).T
revenue2 = area2['Revenue'].values
sal_tol2 = np.tile(area2['Optimum Salinity'], (time, 1)).T

area3 = pd.read_csv('Area3.csv')
crops3 = area3['Crops']
crop_water3 = np.tile(area3['Water requirement'], (time, 1)).T
revenue3 = area3['Revenue'].values
sal_tol3 = np.tile(area3['Optimum Salinity'], (time, 1)).T

area4 = pd.read_csv('Area4.csv')
crops4 = area4['Crops']
crop_water4 = np.tile(area4['Water requirement'], (time, 1)).T
revenue4 = area4['Revenue'].values
sal_tol4 = np.tile(area4['Optimum Salinity'], (time, 1)).T
crops = [crops1, crops2, crops3, crops4]

recharge = pd.read_csv('Recharge.csv').values*10

# Finding the mean of the recharge
r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T

# Creating the covariance matrix and just extracting the diagonal
cov_matrix = np.diag(np.diag(np.cov(recharge, rowvar=False)))

# Finding the Cholesky decomposition
delt = np.linalg.cholesky(cov_matrix)
delta = np.tile(np.array([delt[0, 0], delt[1, 1], delt[2, 2], delt[3, 3]]), (time, 1)).T

t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

land_min = 10
land_max = 500
tww_sal = 1.2
desal_sal = 0.5
domestic_sal = 1

radius = np.round(np.linspace(0, 3.6, 4), 1)
opt_soln = []
prob = []
for v in range(4):
    '''Defining the Model'''
    model = ro.Model()

    # Define decision variables
    q_Sc_1 = model.dvar((len(crops1), time))  # matrix indicating the amount of water from a source allocated to a crop
    q_Sc_2 = model.dvar((len(crops2), time))
    q_Sc_3 = model.dvar((len(crops3), time))
    q_Sc_4 = model.dvar((len(crops4), time))

    q_Wc1 = model.dvar(
        (len(crops1), time))  # vector indicating the amount allocated to the various crops from treated wastewater
    q_Wc2 = model.dvar(
        (len(crops2), time))  # vector indicating the amount allocated to the various crops from treated wastewater
    q_Wc3 = model.dvar(
        (len(crops3), time))  # vector indicating the amount allocated to the various crops from treated wastewater
    q_Wc4 = model.dvar(
        (len(crops4), time))  # vector indicating the amount allocated to the various crops from treated wastewater

    q_Dc1 = model.dvar(
        (len(crops1), time))  # vector indicating the amount allocated to the various crops from desalinated water
    q_Dc2 = model.dvar(
        (len(crops2), time))  # vector indicating the amount allocated to the various crops from desalinated water
    q_Dc3 = model.dvar(
        (len(crops3), time))  # vector indicating the amount allocated to the various crops from desalinated water
    q_Dc4 = model.dvar(
        (len(crops4), time))  # vector indicating the amount allocated to the various crops from desalinated water

    q_S1 = model.dvar(time)
    q_S2 = model.dvar(time)  # vector indicating the amount of water allocated for domestic use for the various sources
    q_S3 = model.dvar(time)  # vector indicating the amount of water allocated for domestic use for the various sources
    q_S4 = model.dvar(time)  # vector indicating the amount of water allocated for domestic use for the various sources

    q_D1 = model.dvar(time)  # the amount of desalinated water allocated for domestic use
    q_D2 = model.dvar(time)  # the amount of desalinated water allocated for domestic use
    q_D3 = model.dvar(time)  # the amount of desalinated water allocated for domestic use
    q_D4 = model.dvar(time)  # the amount of desalinated water allocated for domestic use

    land1 = model.dvar((len(crops1), time))  # the amount of land allocated for the various crops
    land2 = model.dvar((len(crops2), time))
    land3 = model.dvar((len(crops3), time))
    land4 = model.dvar((len(crops4), time))

    # Creating the demand  and recharge uncertainty set
    r_uncertain1 = model.rvar(time)
    r_uncertain2 = model.rvar(time)
    r_uncertain3 = model.rvar(time)
    r_uncertain4 = model.rvar(time)
    r_uncertain = [r_uncertain1, r_uncertain2, r_uncertain3, r_uncertain4]
    d_uncertain1 = model.rvar(time)
    d_uncertain2 = model.rvar(time)
    d_uncertain3 = model.rvar(time)
    d_uncertain4 = model.rvar(time)
    d_uncertain = [d_uncertain1, d_uncertain2, d_uncertain3, d_uncertain4]
    d_set = []
    r_set = []

    for i in range(areas):
        r_set.append((rso.norm(r_uncertain[i]) <= radius[v]))
        d_set.append((d_uncertain[i] >= -1, d_uncertain[i] <= 1))

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
              ((cost[0] * q_Sc_1).sum(axis=0) + (cost[0] * q_S1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) +
               (cost[2] * q_Sc_3).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (cost[3] * q_Sc_4).sum(axis=0) + (cost[3] * q_S4).sum(axis=0)
               + 0.7 * ((q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(axis=0)) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                       q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    '''Defining the constraints'''
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    cum_supply1 = [q_Sc_1.sum(axis=0)[0] + q_S1[0] + q_Dc1.sum(axis=0) + q_D1[0]]
    cum_supply2 = [q_Sc_2.sum(axis=0)[0] + q_S2[0] + q_Dc2.sum(axis=0) + q_D2[0]]
    cum_supply3 = [q_Sc_3.sum(axis=0)[0] + q_S3[0] + q_Dc3.sum(axis=0) + q_D3[0]]
    cum_supply4 = [q_Sc_4.sum(axis=0)[0] + q_S4[0] + q_Dc4.sum(axis=0) + q_D4[0]]
    cum_supply = [cum_supply1, cum_supply2, cum_supply3, cum_supply4]
    recharge_cum = [[r_mean[0, 0]], [r_mean[1, 0]], [r_mean[2, 0]], [r_mean[3, 0]]]
    delta_cum = [[delta[0, 0]], [delta[1, 0]], [delta[2, 0]], [delta[3, 0]]]
    for a in range(0, 4):
        for i in range(1, time):
            cum_supply[a].append(
                cum_supply[a][i - 1] + qSc_vars[a].sum(axis=0)[i] + qS_vars[a][i] + qD_vars[a][i] + qDc_vars[a].sum(axis=0)[
                    i])
            recharge_cum[a].append(recharge_cum[a][i - 1] + r_mean[a, i])
            delta_cum[a].append(delta_cum[a][i - 1] + delta[a, i])
    for i in range(areas):
        # Domestic Demand Constraints
        model.st((qS_vars[i] + qD_vars[i] >= (1 + d_uncertain[i]*error[i]) * water_demand[i]).forall(d_set[i]))

        # Crop Demand Constraints
        model.st(qSc_vars[i] + qDc_vars[i] + qW_vars[i] >= crop_water[i] * land_vars[i])

        # Quality Constraint
        model.st(sal[i] * qS_vars[i] + desal_sal * qD_vars[i] <= domestic_sal * (qS_vars[i] + qD_vars[i]))
        model.st(sal[i] * qSc_vars[i] + tww_sal * qW_vars[i] + desal_sal * qDc_vars[i] <= sal_tol[i] * (
                qSc_vars[i] + qW_vars[i] + qDc_vars[i]))

        # Sources Constraint
        model.st((qW_vars[i].sum(axis=0) <= 0.6 * (1 + d_uncertain[i]*error[i]) * water_demand[i]).forall(d_set[i]))

        # for n in range(time):
        #     model.st(qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= quantity[i] + recharge_cum[n][i] - cum_supply[i][n])
        # model.st((qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i] <= np.tile(quantity[i], time)
        #          + recharge_cum[i] + (delta[i, i] * r_uncertain[i]) - cum_supply[i][1] - 0.1 * np.tile(quantity[i], time)).forall(r_set[i]))
        for n in range(time):
            model.st((qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= 0.9 *
                      (quantity[i])
                      + recharge_cum[i][n] + (delta[i, n] * r_uncertain[i][n]) - cum_supply[i][n]).forall(r_set[i]))

        # Land Constraint
        model.st(land_vars[i] >= land_min)
        model.st(land_vars[i] <= land_max)

    # Non-negativity
    q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3, q_D4,
              q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    for q in q_vars:
        model.st(q >= 0)

    # Solving the model
    model.solve(grb)
#     for i in range(areas):
#         h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(qSc_vars[i].get())], columns=t_set)
#         d1 = pd.DataFrame(data=qSc_vars[i].get(), index=crops[i], columns=t_set)
#         h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(qW_vars[i].get())], columns=t_set)
#         d2 = pd.DataFrame(data=qW_vars[i].get(), index=crops[i], columns=t_set)
#         h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(qDc_vars[i].get())], columns=t_set)
#         d3 = pd.DataFrame(data=qDc_vars[i].get(), index=crops[i], columns=t_set)
#         h_land = pd.DataFrame(index=['Land Allocated', np.sum(land_vars[i].get())], columns=t_set)
#         d4 = pd.DataFrame(data=land_vars[i].get(), index=crops[i], columns=t_set)
#         d5 = pd.DataFrame(data=qD_vars[i].get(), index=t_set, columns=['Desalinated Water'])
#         d6 = pd.DataFrame(data=qS_vars[i].get(), index=t_set, columns=['Groundwater Water'])
#         d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
#             np.sum(qSc_vars[i].get()) + np.sum(qDc_vars[i].get()) + np.sum(qS_vars[i].get()) + np.sum(qD_vars[i].get())]})
#         (pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)).to_excel(os.path.join(folder_path, f'Domestic1 {i + 1} Ω{radius[v]}.xlsx'))
#         (pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)).to_excel(os.path.join(folder_path,
#                                                                                                        f'Crops1 {i + 1} Ω{radius[v]}.xlsx'))
#     opt_soln.append(model.get())
#     prob.append(np.exp(-(v * v)/2))
# df = pd.DataFrame(data=opt_soln, columns=['Optimal Solution'], index=radius)
# df.to_excel(os.path.join(folder_path, 'Optimal Solutionsn.xlsx'))


# Finding the probability of constraint violation


# model.st((qSc_vars[i].sum(axis=0) + qDc_vars[i].sum(axis=0) + qS_vars[i] + qD_vars[i] <= np.tile(quantity[i], time)
#                  + recharge_cum[i] + (delta[i, i] * r_uncertain[i]) - cum_supply[i][1] - 0.1 * np.tile(quantity[i], time)).forall(r_set[i]))
plt.figure()
plt.plot(radius, opt_soln, marker='*')
plt.xlabel('Robustness (Ω)')
plt.ylabel('Revenue ($)')
plt.show()
