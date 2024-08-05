import numpy as np
import pandas as pd
import math
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
years = np.arange(1990, 2021)

aquifer_prod = pd.read_csv('Yearly Production.csv')
train = [0, 0, 14, 0]
yearly_prod = []  # projected yearly aquifer production
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
    yearly_prod.append(projection(aquifer_prod.iloc[train[i-1]:, i-1], years[train[i-1]:], t))  # projection the aquifer production


aquifer = pd.read_csv('Aquifer.csv')
aquifer_sal = pd.read_csv('Yearly Salinity.csv')
sal = aquifer_sal.mean(axis=0)
cost = aquifer['Cost']

"""Finding the recharge from rainfall values"""
rainfall = pd.read_csv('Rainfall.csv').values
area = 490000000  # The area (m2) of land contributing to runoff in each area
beta = 0.1  # Percentage of runoff that ends up as recharge
recharge = np.round(beta * rainfall * 0.001 * area, 0)
yearly_recharge = np.tile(np.round(np.mean(recharge, axis=0), 0), (time, 1)).T

cum_prod = np.cumsum(np.array(yearly_prod), axis=1)  # Yearly production
cum_recharge = np.cumsum(yearly_recharge, axis=1)   # Mean of yearly recharge

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

t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

land_min = 100
land_max = 1500
tww_sal = 1.2
desal_sal = 0.5
domestic_sal = 1


def deterministic():
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
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
               ((cost[0] * q_Sc_1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (cost[2] * q_Sc_3).sum(axis=0) + (
                       cost[3] * q_Sc_4).sum(axis=0) + (
                        cost[0] * q_S1).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (
                        cost[3] * q_S4).sum(axis=0) + 0.7 * (
                        q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(axis=0) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                        q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    for a in range(areas):
        '''Defining the constraints'''
        # Domestic Demand Constraints
        model.st(qS_vars[a] + qD_vars[a] >= water_demand[a])

        # Crop Demand Constraints
        model.st(qSc_vars[a] + qDc_vars[a] + qW_vars[a] >= crop_water[a] * land_vars[a])

        # Quality Constraint
        model.st(sal[a] * qS_vars[a] + desal_sal * qD_vars[a] <= domestic_sal * (qS_vars[a] + qD_vars[a]))
        model.st(sal[a] * qSc_vars[a] + tww_sal * qW_vars[a] + desal_sal * qDc_vars[a] <= sal_tol[a] * (
                qSc_vars[a] + qW_vars[a] + qDc_vars[a]))

        # Sources Constraint
        model.st(qW_vars[a].sum(axis=0) <= 0.6 * water_demand[a])
        model.st(qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 *
                      (yearly_prod[a]) + yearly_recharge[a, :])

        # Land Constraint
        model.st(land_vars[a] >= land_min, land_vars[a] <= land_max)

        # Non-negativity Constraint
        q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
                  q_D4,
                  q_Wc1, q_Wc2, q_Wc3, q_Wc4]
        for v in q_vars:
            model.st(v >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    qW_soln = []
    land_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
        qW_soln.append(qW_vars[s].get())
        land_soln.append(land_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln, qW_soln, land_soln


def rec_unc(ohm):
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
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    # Creating the demand  and recharge uncertainty set
    r_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    r_set = []
    covars = []
    for r in range(areas):
        r_set.append((rso.norm(r_uncertain[r]) <= ohm))
        covars.append(np.cov(aquifer_prod.iloc[r], recharge[r, :], rowvar=False))
    # variance = np.var(recharge, axis=0)    # The variance of recharge for the various areas
    variance = np.var(recharge, axis=0)    # The variance of recharge for the various areas

    delta = []  # Finding the cumulative sum of the Cholesky decomposition of the covariance matrix
    for var in variance:
        delta.append(np.diag(np.diag(np.linalg.cholesky(np.diag(np.full(time, var))))))
    # delta = []  # Finding the cumulative sum of the Cholesky decomposition of the covariance matrix
    # for var in covars:
    #     delta.append(np.linalg.cholesky(var))

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
               ((cost[0] * q_Sc_1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (cost[2] * q_Sc_3).sum(axis=0) + (
                       cost[3] * q_Sc_4).sum(axis=0) + (
                        cost[0] * q_S1).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (
                        cost[3] * q_S4).sum(axis=0) + 0.7 * (
                        q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(axis=0) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                        q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    for a in range(areas):
        '''Defining the constraints'''
        # Domestic Demand Constraints
        model.st(qS_vars[a] + qD_vars[a] >= water_demand[a])

        # Crop Demand Constraints
        model.st(qSc_vars[a] + qDc_vars[a] + qW_vars[a] >= crop_water[a] * land_vars[a])

        # Quality Constraint
        model.st(sal[a] * qS_vars[a] + desal_sal * qD_vars[a] <= domestic_sal * (qS_vars[a] + qD_vars[a]))
        model.st(sal[a] * qSc_vars[a] + tww_sal * qW_vars[a] + desal_sal * qDc_vars[a] <= sal_tol[a] * (
                qSc_vars[a] + qW_vars[a] + qDc_vars[a]))

        # Sources Constraint
        model.st(qW_vars[a].sum(axis=0) <= 0.6 * water_demand[a])
        model.st((qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 *
                      (yearly_prod[a]) + yearly_recharge[a, :] + (delta[a] @ r_uncertain[a])).forall(r_set[a]))
        # model.st((qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 *
        #               (np.tile(np.mean(aquifer_prod.iloc[a]), (time, 1)))
        #               + np.tile(recharge[a, :], (time, 1)) + (delta[a] @ r_uncertain[a]).sum(axis=0)).forall(r_set[a]))

        # Land Constraint
        model.st(land_vars[a] >= land_min, land_vars[a] <= land_max)

        # Non-negativity Constraint
        q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
                  q_D4,
                  q_Wc1, q_Wc2, q_Wc3, q_Wc4]
        for v in q_vars:
            model.st(v >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    qW_soln = []
    land_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
        qW_soln.append(qW_vars[s].get())
        land_soln.append(land_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln, qW_soln, land_soln


def dem_unc(box, ohm):
    # If box uncertainty set, box=1. If not, box = 0 and then we state the radius
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
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    # Creating the uncertainty set
    d_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    d_set = []
    if box == 1:
        for i in range(areas):
            d_set.append((d_uncertain[i] >= -1, d_uncertain[i] <= 1))
    else:
        for i in range(areas):
            d_set.append((rso.norm(d_uncertain[i]) <= ohm))
    rhos = [np.ones((time, time)), np.ones((time, time)), np.ones((time, time)), np.ones((time, time))]
    varis = []
    devs = []
    for k in range(areas):
        var = (error[k] * water_demand[k])/3
        varis.append(var)
        devs.append(np.diag(var**2))
        for r in range(time):
            for j in range(time):
                if r == j:
                    rhos[k][r, j] = 1
            else:
                rhos[k][r, j] = np.exp((j - r - 1) * 0.4)
                rhos[k][j, r] = rhos[k][r, j]

    for l in range(areas):
        for r in range(time):
            for c in range(time):
                if r == c:
                    devs[l][r, c] = devs[l][r, c]
            else:
                devs[l][r, c] = varis[l][r] * varis[l][c]
    deltas = []
    for k in range(areas):
        deltas.append(np.linalg.cholesky(devs[k] * rhos[k]))

    cum_supply = [[0], [0], [0], [0]]
    for a in range(0, 4):
        for i in range(1, time):
            cum_supply[a].append(
                (cum_supply[a][i - 1] + (
                        qSc_vars[a].sum(axis=0)[i - 1] + qS_vars[a][i - 1] + qDc_vars[a].sum(axis=0)[i - 1] + qD_vars[a][
                    i - 1])).sum())

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
               ((cost[0] * q_Sc_1).sum(axis=0) + (cost[0] * q_S1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (
                       cost[1] * q_S2).sum(axis=0) +
                (cost[2] * q_Sc_3).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (cost[3] * q_Sc_4).sum(axis=0) + (
                        cost[3] * q_S4).sum(axis=0)
                + 0.7 * ((q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(
                           axis=0)) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                        q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    for a in range(areas):
        '''Defining the constraints'''
        # Domestic Demand Constraints
        if box == 1:
            model.st((qS_vars[a] + qD_vars[a] >= (1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a]).forall(d_set[a]))
            model.st((qW_vars[a].sum(axis=0) <= 0.6 * (1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a]).forall(d_set[a]))
        else:
            model.st((qS_vars[a] + qD_vars[a] >= (1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a]).forall(d_set[a]))
            model.st((qW_vars[a].sum(axis=0) <= 0.6 * (1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a]).forall(d_set[a]))
            # model.st((qS_vars[a] + qD_vars[a] >= water_demand[a] + d_uncertain[a] @ deltas[a]).forall(d_set[a]))
            # model.st((qW_vars[a].sum(axis=0) <= 0.6 * (water_demand[a] + d_uncertain[a] @ deltas[a])).forall(d_set[a]))
        # Crop Demand Constraints
        model.st(qSc_vars[a] + qDc_vars[a] + qW_vars[a] >= crop_water[a] * land_vars[a])

        # Quality Constraint
        model.st(sal[a] * qS_vars[a] + desal_sal * qD_vars[a] <= domestic_sal * (qS_vars[a] + qD_vars[a]))
        model.st(sal[a] * qSc_vars[a] + tww_sal * qW_vars[a] + desal_sal * qDc_vars[a] <= sal_tol[a] * (
                qSc_vars[a] + qW_vars[a] + qDc_vars[a]))

        # Sources Constraint
        model.st(qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 *
                      (yearly_prod[a]) + yearly_recharge[a, :])
        # for n in range(time):
        #     model.st((qSc_vars[a].sum(axis=0)[n] + qDc_vars[a].sum(axis=0)[n] + qS_vars[a][n] + qD_vars[a][n] <= 0.9 *
        #               (cum_prod[a, n]) + cum_recharge[a, n] - cum_supply[a][n]))

        # Land Constraint
        model.st(land_vars[a] >= land_min)
        model.st(land_vars[a] <= land_max)

        # Non-negativity Constraint
        q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
                  q_D4,
                  q_Wc1, q_Wc2, q_Wc3, q_Wc4]
        for v in q_vars:
            model.st(v >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    qW_soln = []
    land_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
        qW_soln.append(qW_vars[s].get())
        land_soln.append(land_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln, qW_soln, land_soln


def dem_rec_unc(box, ohm):
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
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    # Creating the demand  and recharge uncertainty set
    r_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    d_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    d_set = []
    r_set = []
    if box == 1:
        for i in range(areas):
            r_set.append((rso.norm(r_uncertain[i]) <= ohm))
            d_set.append((d_uncertain[i] >= -1, d_uncertain[i] <= 1))
    else:
        for i in range(areas):
            r_set.append((rso.norm(r_uncertain[i]) <= ohm))
            d_set.append((rso.norm(d_uncertain[i]) <= ohm))

    variance = np.var(recharge, axis=0)    # The variance of recharge for the various areas

    delta = []  # Finding the cumulative sum of the Cholesky decomposition of the covariance matrix
    for var in variance:
        delta.append(np.diag(np.diag(np.linalg.cholesky(np.diag(np.full(time, var))))))
    cum_supply = [[0], [0], [0], [0]]
    for a in range(0, 4):
        for i in range(1, time):
            cum_supply[a].append(
                (cum_supply[a][i - 1] + (
                        qSc_vars[a].sum(axis=0)[i - 1] + qS_vars[a][i - 1] + qDc_vars[a].sum(axis=0)[i - 1] + qD_vars[a][
                    i - 1])).sum())

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
               ((cost[0] * q_Sc_1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (cost[2] * q_Sc_3).sum(axis=0) + (
                       cost[3] * q_Sc_4).sum(axis=0) + (
                        cost[0] * q_S1).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (
                        cost[3] * q_S4).sum(axis=0) + 0.7 * (
                        q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(axis=0) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                        q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    for a in range(areas):
        '''Defining the constraints'''
        # Domestic Demand Constraints
        model.st((qS_vars[a] + qD_vars[a] >= (1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a]).forall(d_set[a]))

        # Crop Demand Constraints
        model.st(qSc_vars[a] + qDc_vars[a] + qW_vars[a] >= crop_water[a] * land_vars[a])

        # Quality Constraint
        model.st(sal[a] * qS_vars[a] + desal_sal * qD_vars[a] <= domestic_sal * (qS_vars[a] + qD_vars[a]))
        model.st(sal[a] * qSc_vars[a] + tww_sal * qW_vars[a] + desal_sal * qDc_vars[a] <= sal_tol[a] * (
                qSc_vars[a] + qW_vars[a] + qDc_vars[a]))

        # Sources Constraint
        model.st((qW_vars[a].sum(axis=0) <= 0.6 * ((1 + d_uncertain[a] * math.sqrt(error[a])) * water_demand[a])).forall(d_set[a]))
        model.st((qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 *
                      (yearly_prod[a])
                      + yearly_recharge[a, :] + (delta[a] @ r_uncertain[a])).forall(r_set[a]))
        # for z in range(time):
        #     model.st((qSc_vars[a].sum(axis=0)[z] + qDc_vars[a].sum(axis=0)[z] + qS_vars[a][z] + qD_vars[a][z] <= 0.9 *
        #               (cum_prod[a, z])
        #               + cum_recharge[a, z] + (delta[a][:, z] @ r_uncertain[a]) - cum_supply[a][z]).forall(r_set[a]))

        # Land Constraint
        model.st(land_vars[a] >= land_min, land_vars[a] <= land_max)

        # Non-negativity Constraint
        q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
                  q_D4,
                  q_Wc1, q_Wc2, q_Wc3, q_Wc4]
        for v in q_vars:
            model.st(v >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    qW_soln = []
    land_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
        qW_soln.append(qW_vars[s].get())
        land_soln.append(land_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln, qW_soln, land_soln


def prob_vio(q1, q2, q3, q4, rs):
    """This function calculates the probability of violating the constraints"""
    prob = []
    for p in range(1000):
        proba = []
        for a in range(4):
            proba.append(
                    (np.sum(q1[a], axis=0) + np.sum(q2[a], axis=0) + q3[a] + q4[a] > 0.9 * (yearly_prod[a]) + rs[a][:, p]))
        prob.append(np.sum(proba) > 1)   # This gives a sum of all falses.
    return sum(prob)/len(prob)


def prob_vio_demrec(q1, q2, q3, q4, q5,  rs):
    """This function calculates the probability of violating the constraints"""
    dem_mat = []
    for r in range(areas):
        dem_vec = []
        for w in range(time):
            dem_vec.append(np.random.normal(water_demand[r][w], (math.sqrt(error[r]))/3 * water_demand[r][w], 1000))
        dem_mat.append(np.array(dem_vec))
    prob = []
    for p in range(1000):
        proba = []
        proba1 = []
        for a in range(4):
            proba.append(
                    (np.sum(q1[a], axis=0) + np.sum(q2[a], axis=0) + q3[a] + q4[a] > 0.9 * (yearly_prod[a]) + rs[a][:, p]))
            proba1.append((q3[a] + q4[a] < dem_mat[a][:, p]))  # If 1, then it means false and a sum will be falses
            proba1.append(np.sum(q5[a], axis=0) > 0.6 * dem_mat[a][:, p])
        prob.append(np.sum(proba) + np.sum(proba1) > 1)   # This gives a sum of all falses.
    return sum(prob)/len(prob)


def prob_vio_dem(q3, q4, q5):
    """This function calculates the probability of violating the constraints"""
    dem_mat = []
    for r in range(areas):
        dem_vec = []
        for w in range(time):
            dem_vec.append(np.random.normal(water_demand[r][w], (math.sqrt(error[r]))/3 * water_demand[r][w], 1000))
        dem_mat.append(np.array(dem_vec))
    prob = []
    for p in range(1000):
        proba1 = []
        for a in range(4):
            proba1.append((q3[a] + q4[a] < dem_mat[a][:, p]))  # If 1, then it means false and a sum will be falses
            proba1.append(np.sum(q5[a], axis=0) > 0.6 * dem_mat[a][:, p])
        prob.append(np.sum(proba1) > 1)   # This gives a sum of all falses.
    return sum(prob)/len(prob)


# Considering uncertainties in salinity levels
def dem_recq_unc(ohm):
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
    qD_vars = [q_D1, q_D2, q_D3, q_D4]
    qS_vars = [q_S1, q_S2, q_S3, q_S4]
    qW_vars = [q_Wc1, q_Wc2, q_Wc3, q_Wc4]
    qDc_vars = [q_Dc1, q_Dc2, q_Dc3, q_Dc4]
    qSc_vars = [q_Sc_1, q_Sc_2, q_Sc_3, q_Sc_4]
    sal_tol = [sal_tol1, sal_tol2, sal_tol3, sal_tol4]
    crop_water = [crop_water1, crop_water2, crop_water3, crop_water4]
    land_vars = [land1, land2, land3, land4]

    # Creating the demand  and recharge uncertainty set
    r_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    d_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    q_uncertain = [model.rvar(time), model.rvar(time), model.rvar(time), model.rvar(time)]
    d_set = []
    r_set = []
    q_set = []
    for i in range(areas):
        r_set.append((rso.norm(r_uncertain[i]) <= ohm))
        d_set.append((rso.norm(d_uncertain[i]) <= ohm))
        q_set.append((rso.norm(q_uncertain[i]) <= ohm))

    cum_supply = [[0], [0], [0], [0]]
    for a in range(0, 4):
        for i in range(1, time):
            cum_supply[a].append(
                (cum_supply[a][i - 1] + (
                        qSc_vars[a].sum(axis=0)[i-1] + qS_vars[a][i-1] + qDc_vars[a].sum(axis=0)[i-1] + qD_vars[a][i-1])).sum())

    variance = np.var(recharge, axis=0)    # The variance of recharge for the various areas

    delta = []  # Finding the cumulative sum of the Cholesky decomposition of the covariance matrix
    for var in variance:
        delta.append(np.diag(np.cumsum(np.diag(np.linalg.cholesky(np.diag(np.full(time, var)))))))

    # Objective Function
    model.max(((revenue1.T @ land1).sum(axis=0) + (revenue2.T @ land2).sum(axis=0) + (revenue3.T @ land3).sum(axis=0) + (
            revenue4.T @ land4).sum(axis=0) -
               ((cost[0] * q_Sc_1).sum(axis=0) + (cost[1] * q_Sc_2).sum(axis=0) + (cost[2] * q_Sc_3).sum(axis=0) + (
                       cost[3] * q_Sc_4).sum(axis=0) + (
                        cost[0] * q_S1).sum(axis=0) + (cost[1] * q_S2).sum(axis=0) + (cost[2] * q_S3).sum(axis=0) + (
                        cost[3] * q_S4).sum(axis=0) + 0.7 * (
                        q_Dc1.sum(axis=0) + q_Dc2.sum(axis=0) + q_Dc3.sum(axis=0) + q_Dc4.sum(axis=0) + q_D1 + q_D2 + q_D3 + q_D4) + 0.4 * (
                        q_Wc1.sum(axis=0) + q_Wc2.sum(axis=0) + q_Wc3.sum(axis=0) + q_Wc4.sum(axis=0)))).sum())

    for a in range(areas):
        '''Defining the constraints'''
        # Domestic Demand Constraints
        model.st((qS_vars[a] + qD_vars[a] >= (1 + d_uncertain[a] * error[a]) * water_demand[a]).forall(d_set[a]))

        # Crop Demand Constraints
        model.st(qSc_vars[a] + qDc_vars[a] + qW_vars[a] >= crop_water[a] * land_vars[a])

        # Quality Constraint
        model.st(sal[a] * qS_vars[a] + desal_sal * qD_vars[a] <= domestic_sal * (qS_vars[a] + qD_vars[a]))
        model.st(sal[a] * qSc_vars[a] + tww_sal * qW_vars[a] + desal_sal * qDc_vars[a] <= sal_tol[a] * (
                qSc_vars[a] + qW_vars[a] + qDc_vars[a]))

        # Sources Constraint
        model.st((qW_vars[a].sum(axis=0) <= 0.6 * ((1 + d_uncertain[a] * error[a]) * water_demand[a])).forall(d_set[a]))
        for z in range(time):
            model.st((qSc_vars[a].sum(axis=0)[z] + qDc_vars[a].sum(axis=0)[z] + qS_vars[a][z] + qD_vars[a][z] <= 0.9 *
                      (cum_prod[a, z])
                      + cum_recharge[a, z] + (delta[a][:, z] @ r_uncertain[a]) - cum_supply[a][z]).forall(r_set[a]))

        # Land Constraint
        model.st(land_vars[a] >= land_min, land_vars[a] <= land_max)

        # Non-negativity Constraint
        q_vars = [q_Sc_1, q_S1, q_Sc_2, q_S2, q_Sc_3, q_S3, q_Sc_4, q_S4, q_Dc1, q_Dc2, q_Dc3, q_Dc4, q_D1, q_D2, q_D3,
                  q_D4,
                  q_Wc1, q_Wc2, q_Wc3, q_Wc4]
        for v in q_vars:
            model.st(v >= 0)

    # Solving the model
    model.solve(grb)
    qS_soln = []
    qSc_soln = []
    qDc_soln = []
    qD_soln = []
    qW_soln = []
    land_soln = []
    for s in range(areas):
        qS_soln.append(qS_vars[s].get())
        qSc_soln.append(qSc_vars[s].get())
        qDc_soln.append(qDc_vars[s].get())
        qD_soln.append(qD_vars[s].get())
        qW_soln.append(qW_vars[s].get())
        land_soln.append(land_vars[s].get())
    return model.get(), qSc_soln, qDc_soln, qS_soln, qD_soln, qW_soln, land_soln
