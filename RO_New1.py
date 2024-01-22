import numpy as np
import pandas as pd
import os
from rsome import ro
from Functions import MAPE, projection, linear_model, linear_param
import matplotlib.pyplot as plt

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

t_set = []
for i in range(1, time + 1):
    t_set.append(f't{i}')

aquifer = pd.read_csv('Aquifer.csv')
quantity = aquifer['Quantity'] * 1
sal = aquifer['Salinity']
cost = aquifer['Cost']
recharge = np.tile(aquifer['Recharge'], (time, 1)).T * 1

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

# water_demand = np.round(pd.read_csv('Water_demand.csv').values, 0)
land_min = 5
land_max = 500
tww_sal = 1.2
desal_sal = 0.25
domestic_sal = 0.5
crops = [crops1, crops2, crops3, crops4]
revenue = [revenue1, revenue2, revenue3, revenue4]

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
for a in range(0, 4):
    for i in range(1, time):
        cum_supply[a].append(
            (cum_supply[a][i - 1] + (
                    qSc_vars[a].sum(axis=0)[i] + qS_vars[a][i] + qDc_vars[a].sum(axis=0)[i] + qD_vars[a][i])).sum())
        recharge_cum[a].append(recharge_cum[a][i - 1] + re_mean[a])


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

    # model.st(qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a] <= 0.9 * np.tile(quantity[a], time) +
    #          recharge_cum[a] - cum_supply[a][19])
    for n in range(time):
        model.st((qSc_vars[a].sum(axis=0)[n] + qDc_vars[a].sum(axis=0)[n] + qS_vars[a][n] + qD_vars[a][n] <= 0.9 *
                  (quantity[a])
                  + recharge_cum[a][n] - cum_supply[a][n]))

    # model.st((qSc_vars[a].sum(axis=0) + qDc_vars[a].sum(axis=0) + qS_vars[a] + qD_vars[a]).sum() <= 0.9 * (quantity[a, 0]) + np.sum(recharge[0, 0:time]))
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
model.solve()

# Printing the optimal solution
print(f"Optimal Solution: {model.get()}")

folder_path = 'C:/Users/User/OneDrive - BGU/Documents/Life at BGU/Research Work/PycharmProjects/pythonProject/venv/Scripts/RO_New/Deterministic Results_final'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
plt.rcParams['font.size'] = '14'
for i in range(areas):
    h_aquifer = pd.DataFrame(index=['Brackish Groundwater', np.sum(qSc_vars[i].get())], columns=t_set)
    d1 = pd.DataFrame(data=qSc_vars[i].get()/1000000, index=crops[i], columns=t_set)
    h_tww = pd.DataFrame(index=['Treated Wastewater', np.sum(qW_vars[i].get())], columns=t_set)
    d2 = pd.DataFrame(data=qW_vars[i].get()/1000000, index=crops[i], columns=t_set)
    h_desal = pd.DataFrame(index=['Desalinated Water', np.sum(qDc_vars[i].get())], columns=t_set)
    d3 = pd.DataFrame(data=qDc_vars[i].get()/1000000, index=crops[i], columns=t_set)
    h_land = pd.DataFrame(index=['Land Allocated', np.sum(land_vars[i].get())], columns=t_set)
    d4 = pd.DataFrame(data=land_vars[i].get(), index=crops[i], columns=t_set)
    d5 = pd.DataFrame(data=qD_vars[i].get(), index=t_set, columns=['Desalinated Water'])
    d6 = pd.DataFrame(data=qS_vars[i].get(), index=t_set, columns=['Groundwater Water'])
    d7 = pd.DataFrame({'Total Amount of water from the aquifer used': [
        np.sum(qSc_vars[i].get()) + np.sum(qDc_vars[i].get()) + np.sum(qS_vars[i].get()) + np.sum(qD_vars[i].get())]})
    dome = pd.concat([d5, d6], axis=1)
    domestic = pd.concat([pd.concat([d5, d6], axis=1), d7], axis=1)
    d = pd.concat([h_aquifer, d1, h_tww, d2, h_desal, d3, h_land, d4], axis=0)
    domestic.to_excel(os.path.join(folder_path, f'Domestic Use for Area {i + 1}.xlsx'))
    d.to_excel(os.path.join(folder_path, f'Crops Output for Area {i + 1}.xlsx'))

    # Calculating the Salinity of water allocated to the various crops
    sal_allo = (sal[i] * qSc_vars[i].get() + tww_sal*qW_vars[i].get() + desal_sal * qDc_vars[i].get())/(qSc_vars[i].get() + qW_vars[i].get() + qDc_vars[i].get())
    d_sal_allo = pd.DataFrame(data=sal_allo, index=crops[i], columns=t_set)

    # Plotting the salinity of water allocated for the crops
    for c in range(len(crops[i])):
        plt.plot(t_set, d_sal_allo.iloc[c], label=f'{crops[i][c]}', linewidth=2)
    plt.ylabel('Salinity of Water Allocated (dS/m)')
    plt.xlabel('Timestep')
    plt.legend()
    plt.title(f'Area {i+1}', fontsize=20)
    plt.show()

    # Plotting the results for water allocated to the crops
    fig, axes = plt.subplots(3, sharex=True, figsize=[11, 9])
    for c in range(len(crops[i])):
        axes[0].plot(t_set, d1.iloc[c], label=f'{crops[i][c]}', linewidth=2)
        axes[1].plot(t_set, d3.iloc[c], label=f'{crops[i][c]}', linewidth=2)
        axes[2].plot(t_set, d2.iloc[c], label=f'{crops[i][c]}', linewidth=2)
    axes[0].set_title('Brackish Groundwater')
    axes[1].set_title('Desalinated Water')
    axes[2].set_title('Treated Wastewater')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.135, 3.45), fancybox=True, shadow=True, fontsize=12)
    plt.legend(ncol=7, loc='upper right', bbox_to_anchor=(1, -0.25), fancybox=True, fontsize=12)
    fig.add_subplot(1, 1, 1, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.ylabel('Water Allocated (x10$^6$ m$^3$)')
    plt.xlabel('Timestep')
    fig.suptitle(f'Area {i+1}', fontsize=20)
    plt.show()
    #
    # A bar chart for the water allocated
    width = 0.35
    fig = plt.subplots(figsize=(10, 7))
    p1 = plt.bar(crops[i], d1.sum(axis=1), width, color='r')
    p2 = plt.bar(crops[i], d3.sum(axis=1), width, bottom=d1.sum(axis=1), color='b')
    p3 = plt.bar(crops[i], d2.sum(axis=1), width, bottom=d3.sum(axis=1)+d1.sum(axis=1), color='g')
    plt.legend((p1[0], p2[0], p3[0]), ('Brackish Groundwater', 'Desalinated Water', 'Treated Wastewater'))
    plt.title(f'Area {i+1}')
    plt.ylabel('Water Allocated (x10$^6$ m$^3$)')
    plt.show()
    #
    width = 0.35
    # fig = plt.subplots(figsize=(10, 7))
    # p1 = plt.bar(crops[i], d1.mean(axis=1), width, yerr=d1.std(axis=1))
    # p2 = plt.bar(crops[i], d3.mean(axis=1), width, bottom=d1.mean(axis=1), yerr=d3.std(axis=1))
    # p3 = plt.bar(crops[i], d2.mean(axis=1), width, bottom=d3.mean(axis=1)+d1.mean(axis=1), yerr=d2.std(axis=1))
    # plt.legend((p1[0], p2[0], p3[0]), ('Brackish Groundwater', 'Desalinated Water', 'Treated Wastewater'))
    # plt.title(f'Area {i+1}')
    # plt.ylabel('Water Allocated (x10$^6$ m$^3$)')
    # plt.show()
    #
    # Plotting the results for land allocated
    for c in range(len(crops[i])):
        plt.plot(t_set, d4.iloc[c], label=f'{crops[i][c]}', linewidth=2)
    plt.ylabel('Land Allocated (hectares)')
    plt.xlabel('Timestep')
    plt.legend()
    plt.title(f'Area {i+1}', fontsize=20)
    plt.show()
    #
    # A bar chart for the land allocated
    plt.bar(t_set, d4.sum(axis=0), width, color='g')
    plt.ylabel('Total Land Allocated (hectares)')
    plt.title(f'Area {i+1}')
    plt.show()

    plt.bar(crops[i], d4.sum(axis=1), width=0.2, color='m')
    plt.ylabel('Land Allocated (hectares)')
    plt.title(f'Area {i+1}')
    plt.show()
    # # Plotting a graph for the domestic use
    # plt.plot(t_set, dome.iloc[:, 0], label='Desalinated Water', linewidth=2)
    # plt.plot(t_set, dome.iloc[:, 1], label='Brackish Groundwater', linewidth=2)
    # plt.title(f'Domestic Use for Area {i+1}')
    # plt.ylabel('Water Allocated (x10$^6$ m$^3$)')
    # plt.show()


