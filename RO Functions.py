import numpy as np
import pandas as pd
import os
from numpy.linalg import inv
import rsome as rso
from rsome import ro, grb_solver as grb
from Functions import MAPE, projection, linear_model, linear_param

pd.set_option('display.max_columns', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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
for i in range(1, areas + 1):
    gdp_new = np.hstack((gdp.iloc[a:, 1], projection(gdp.iloc[a:, 1], year, t)))
    pop_new = np.hstack((population.iloc[a:, i], projection(population.iloc[a:, i], year, t)))
    p1 = linear_param(water_consumption.iloc[a:7, i], gdp.iloc[a:7, 1], population.iloc[a:7, i])
    predict_water = np.round(linear_model(gdp_new, pop_new, p1), 0)
    water_demand.append(predict_water[8:])
    mape = MAPE(water_consumption.iloc[:, i], predict_water[0:len(water_consumption.iloc[:, i])])
    error.append((mape / 100))

aquifer = pd.read_csv('Aquifer.csv')
quantity = aquifer['Quantity'] * 1
sal = aquifer['Salinity']
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

recharge = pd.read_csv('Recharge.csv').values

# Finding the mean of the recharge
r_mean = np.tile(np.mean(recharge, axis=0), (time, 1)).T

# Creating the covariance matrix and just extracting the diagonal
cov_matrix = np.diag(np.diag(np.cov(recharge, rowvar=False)))

# Finding the Cholesky decomposition
delta = np.linalg.cholesky(cov_matrix)

t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

# water_demand = np.round(pd.read_csv('Water_demand.csv').values, 0)
land_min = 5
land_max = 500
tww_sal = 1.2
desal_sal = 0.5
domestic_sal = 1


def dem_rec_unc(ohm):
    """Defining the Model"""
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
    for a in range(areas):
        r_set.append((rso.norm(r_uncertain[a]) <= ohm))
        d_set.append((d_uncertain[a] >= -1, d_uncertain[a] <= 1))

    # Objective Function
    model.max(
        ((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
                revenue4.T @ land4).sum(axis=0) -
         ((cost[0] * q_Sc_1).sum(axis=0) + (cost[0] * q_S1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (
                     cost[1] * q_S2).sum(axis=0) +
          (cost[2] * q_Sc_3).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (cost[3] * q_Sc_4).sum(axis=0) + (
                      cost[3] * q_S4).sum(axis=0)
          + 0.7 * ((q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(
                     axis=0)) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
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

    cum_supply1 = [(q_Sc_1.sum(axis=0)[0] + q_S1[0] + q_Dc1.sum(axis=0) + q_D1[0]).sum()]
    cum_supply2 = [(q_Sc_2.sum(axis=0)[0] + q_S2[0] + q_Dc2.sum(axis=0) + q_D2[0]).sum()]
    cum_supply3 = [(q_Sc_3.sum(axis=0)[0] + q_S3[0] + q_Dc3.sum(axis=0) + q_D3[0]).sum()]
    cum_supply4 = [(q_Sc_4.sum(axis=0)[0] + q_S4[0] + q_Dc4.sum(axis=0) + q_D4[0]).sum()]
    cum_supply = [cum_supply1, cum_supply2, cum_supply3, cum_supply4]
    re_mean = np.mean(recharge, axis=0)
    recharge_cum = [[re_mean[0]], [re_mean[1]], [re_mean[2]], [re_mean[3]]]
    for m in range(0, 4):
        for i in range(1, time):
            cum_supply[m].append(
                (cum_supply[m][i - 1] + (
                        qSc_vars[m].sum(axis=0)[i] + qS_vars[m][i] + qDc_vars[m].sum(axis=0)[i] + qD_vars[m][i])).sum())
            recharge_cum[m].append(recharge_cum[m][i - 1] + re_mean[m])

    for n in range(areas):
        # Domestic Demand Constraints
        model.st((qS_vars[n] + qD_vars[n] >= (1 + d_uncertain[n] * error[n]) * water_demand[n]).forall(d_set[n]))

        # Crop Demand Constraints
        model.st(qSc_vars[n] + qDc_vars[n] + qW_vars[n] >= crop_water[n] * land_vars[n])

        # Quality Constraint
        model.st(sal[n] * qS_vars[n] + desal_sal * qD_vars[n] <= domestic_sal * (qS_vars[n] + qD_vars[n]))
        model.st(sal[n] * qSc_vars[n] + tww_sal * qW_vars[n] + desal_sal * qDc_vars[n] <= sal_tol[n] * (
                qSc_vars[n] + qW_vars[n] + qDc_vars[n]))

        # Sources Constraint
        model.st((qW_vars[n].sum(axis=0) <= 0.6 * (1 + d_uncertain[n] * error[n]) * water_demand[n]).forall(d_set[n]))

        # for n in range(time):
        #     model.st(qSc_vars[i].sum(axis=0)[n] + qDc_vars[i].sum(axis=0)[n] + qS_vars[i][n] + qD_vars[i][n] <= quantity[i] + recharge_cum[n][i] - cum_supply[i][n])
        for z in range(time):
            model.st((qSc_vars[n].sum(axis=0)[z] + qDc_vars[n].sum(axis=0)[z] + qS_vars[n][z] + qD_vars[n][z] <= 0.9 *
                      quantity[n]
                      + recharge_cum[n][z] + (delta[n, n] * r_uncertain[n][z]) - cum_supply[n][z]).forall(r_set[n]))

        # Land Constraint
        model.st(land_vars[n] >= land_min)
        model.st(land_vars[n] <= land_max)

    # Non-negativity
    q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
              q_D4,
              q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    for q in q_vars:
        model.st(q >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln


# for z in range(time):
#           model.st((qSc_vars[n].sum(axis=0)[z] + qDc_vars[n].sum(axis=0)[z] + qS_vars[n][z] + qD_vars[n][z] <= 0.9 *
#                     quantity[n]
#                     + recharge_cum[n][z] + (delta[n, n] * r_uncertain[n][z]) - cum_supply[n][z]).forall(r_set[n]))
re_mean = np.mean(recharge, axis=0)
re_std = np.std(recharge, axis=0)
rs = []
for r in range(areas):
    rs.append(np.random.normal(re_mean[r], re_std[r], (time, 1000)))
print(np.cumsum(rs[1], axis=1))

def prob_vio(q1, q2, q3, q4, rs):
    cum_supply1 = [np.sum(q1[0], axis=0)[0] + np.sum(q2[0], axis=0)[0] + q3[0][0] + q4[0][0]]
    cum_supply2 = [np.sum(q1[1], axis=0)[0] + np.sum(q2[1], axis=0)[0] + q3[1][0] + q4[1][0]]
    cum_supply3 = [np.sum(q1[2], axis=0)[0] + np.sum(q2[2], axis=0)[0] + q3[2][0] + q4[2][0]]
    cum_supply4 = [np.sum(q1[3], axis=0)[0] + np.sum(q2[3], axis=0)[0] + q3[3][0] + q4[3][0]]
    cum_supply = [cum_supply1, cum_supply2, cum_supply3, cum_supply4]
    re_mean = np.mean(recharge, axis=0)
    recharge_cum = [[re_mean[0]], [re_mean[1]], [re_mean[2]], [re_mean[3]]]
    for m in range(0, 4):
        for i in range(1, time):
            cum_supply[m].append(cum_supply[m][i - 1] + np.sum(q1[m], axis=0)[i] + np.sum(q2[m], axis=0)[i] + q3[m][i] + q4[m][i])
            recharge_cum[m].append(recharge_cum[m][i - 1] + re_mean[m])
    prob = []
    cum1 = np.cumsum(rs[0], axis=1)
    cum2 = np.cumsum(rs[1], axis=1)
    cum3 = np.cumsum(rs[2], axis=1)
    cum4 = np.cumsum(rs[3], axis=1)

    for p in range(1000):
        for a in range(4):
            prob.append(np.mean(np.sum(q1[0], axis=0) + np.sum(q2[0], axis=0) + q3[0] + q4[0] < 0.9 * np.tile(quantity[a], time) + recharge_cum[a] - cum_supply[a]))

    return prob
