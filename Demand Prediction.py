import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import AGMC_13, projection, AGMC_12a, grey_degree2, MAPE, linear_model, linear_param, ANN
from sklearn.metrics import mean_squared_error


pd.set_option('display.max_columns', 999)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

'''Importing the data'''
population = pd.read_csv('Population.csv')
gdp = pd.read_csv('GDP.csv')
water_consumption = pd.read_csv('Domestic Water Consumption.csv')
a = 0
year = np.array(population.iloc[a:, 0])

'''Predicting Water Consumption for the different Areas'''

time = 20
t = 2020 + time
areas = 4
year_new = np.hstack((year, np.arange(year[-1]+1, t+1)))

# g = pd.DataFrame({'Factors': ['Population', 'GDP', 'GDP per capita', 'Cost of Water']})
# for i in range(1, areas+1):
#     gdp_grey = grey_degree2(water_consumption.iloc[:, i], gdp.iloc[:, 1])
#     pop_grey = grey_degree2(water_consumption.iloc[:, i], population.iloc[:, i])
#     gdpp_grey = grey_degree2(water_consumption.iloc[:, i], gdp.iloc[:, 2])
#     cost_grey = grey_degree2(water_consumption.iloc[:, i], gdp.iloc[:, 3])
#     df = [pop_grey, gdp_grey, gdpp_grey, cost_grey]
#     g.insert(loc=len(g.columns), column=f'Area {i}', value=df)
# g.to_excel('Grey Relational Analysis.xlsx')


d1 = pd.DataFrame(year_new, columns=['Year'])
opt_param_pop = [0.009, 0.0001, 0.0002, 0.01]
# opt_param_pop = [0, 0.009, 0.08, 0.01]
# ok1 = [0.009(for2), 0.086(for3)] 0.0005, 0.095
# error = pd.DataFrame({'Model': ['AGMC_12', 'Regression Model', 'ANN']})#, 'AGMC_13']})#, 'ANN']})

for i in range(1, areas+1):
    water_consumption_train = np.array(water_consumption.iloc[a:7, i])
    gdp_new = np.hstack((gdp.iloc[a:, 1], projection(gdp.iloc[a:, 1], year, t)))
    pop_new = np.hstack((population.iloc[a:, i], projection(population.iloc[a:, i], year, t)))
    predict_water1 = (np.array(AGMC_12a(water_consumption_train, pop_new, opt_param_pop[i-1])))/1000
    p1 = linear_param(water_consumption.iloc[a:7, i], gdp.iloc[a:7, 1], population.iloc[a:7, i])
    predict_water2 = linear_model(gdp_new, pop_new, p1)/1000
    # predict_water4 = (np.array(AGMC_13(water_consumption_train[:4], pop_new, gdp_new, opt_param_pop[i])))/1000
    # predict_water3 = np.array(ANN(water_consumption_train, gdp_new, pop_new, 100))/1000
    # Creating a DataFrame
    d1.insert(loc=len(d1.columns), column=f'AGMC12(Pop) Area {i}', value=predict_water1)
    d1.insert(loc=len(d1.columns), column=f'Regression(GDP) Area {i}', value=predict_water2)
    # d1.insert(loc=len(d1.columns), column=f'ANN Area {i}', value=predict_water3)

    # d1.insert(loc=len(d1.columns), column=f'ANN Area {i}', value=predict_water3)
    # mape1 = MAPE(water_consumption.iloc[:, i]/1000, predict_water1[0:len(water_consumption.iloc[:, i])])
    # mape2 = MAPE(water_consumption.iloc[:, i]/1000, predict_water2[0:len(water_consumption.iloc[:, i])])
    # # mape3 = MAPE(water_consumption.iloc[:, i]/1000, predict_water3[0:len(water_consumption.iloc[:, i])])
    # # mape4 = MAPE(water_consumption.iloc[:, i]/1000, predict_water4[0:len(water_consumption.iloc[:, i])])
    # mse1 = mean_squared_error(water_consumption.iloc[:, i]/1000, predict_water1[0:len(water_consumption.iloc[:, i])])
    # mse2 = mean_squared_error(water_consumption.iloc[:, i]/1000, predict_water2[0:len(water_consumption.iloc[:, i])])
    # mse4 = mean_squared_error(water_consumption.iloc[:, i]/1000, predict_water4[0:len(water_consumption.iloc[:, i])])
    # mse3 = mean_squared_error(water_consumption.iloc[:, i]/1000, predict_water3[0:len(water_consumption.iloc[:, i])])
    # df1 = [mape1, mape2, mape3]
    # df2 = [mse1, mse2, mse3]
    # error.insert(loc=len(error.columns), column=f'MAPE Area {i}', value=df1)
    # error.insert(loc=len(error.columns), column=f'MSE Area {i}', value=df2)
    plt.figure()
    plt.rcParams['font.size'] = '16'
    plt.plot(year_new[0:len(water_consumption.iloc[a:, i])], water_consumption.iloc[a:, i]/1000, label='Actual Water Consumption', color='c', marker='*', linewidth=2)
    plt.plot(year_new[0:len(water_consumption.iloc[a:, i])], predict_water1[0:len(water_consumption.iloc[a:, i])], label='Modelling Values by AGMC(1,N) Model', color='g', marker='^', linewidth=2)
    plt.plot(year_new[0:len(water_consumption.iloc[a:, i])], predict_water2[0:len(water_consumption.iloc[a:, i])], label='Modelling Values by MLR Model', color='r', marker='o', linewidth=2)
    # plt.plot(year_new[0:len(water_consumption.iloc[a:, i])], predict_water4[0:len(water_consumption_train)], label='Modelling Values by AGMC13 Model', color='b')
    # plt.plot(year_new[0:len(water_consumption.iloc[a:, i])], predict_water3[0:len(water_consumption.iloc[a:, i])], label='Modelling Values by ANN Model', color='m', marker='s')
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Water Consumption (x10$^3$ m$^3$)', fontsize=20)
    plt.title(f'Area {i}', fontsize=20)
    plt.xticks()
    plt.yticks()
    plt.legend()
    plt.show()

    # plt.figure()
    # plt.plot(year_new[8:], predict_water1[8:], label='Modelling Values by AGMC(1,N) Model', color='g', marker='^')
    # plt.plot(year_new[8:], predict_water2[8:], label='Modelling Values by MLR Model', color='r', marker='o')
    # plt.plot(year_new[8:], predict_water3[8:], label='Modelling Values by ANN Model', color='m', marker='s')
    # plt.xlabel('Year', fontsize=14)
    # plt.ylabel('Water Consumption (*10^3 cubic meters)', fontsize=14)
    # plt.title(f'Area{i}', fontsize=16)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.legend(fontsize=12)
    # plt.show()
# error.to_excel('Error for Demand Prediction.xlsx')
# d1.to_excel('Demand Prediction Results_final.xlsx')


#     predict_water2 = AGMC_12a(water_consumption.iloc[:, i], gdp_new, 0.001)
#     predict_water3 = AGMC_13(water_consumption.iloc[:, i], gdp_new, pop_new, 0.0069)
#     p = linear_param(water_consumption.iloc[:, i], gdp.iloc[:, 1], population.iloc[:, i])
#     predict_water4 = linear_model(gdp_new, pop_new, p)
#     predict_water5 = ANN(water_consumption.iloc[:, i], gdp_new, pop_new, 200)
#     # Creating a DataFrame
#     d1.insert(loc=len(d1.columns), column=f'AGMC12(Pop) Area {i}', value=predict_water1)
#     d1.insert(loc=len(d1.columns), column=f'AGMC12(GDP) Area {i}', value=predict_water2)
#     d1.insert(loc=len(d1.columns), column=f'AGMC13 Area {i}', value=predict_water3)
#     d1.insert(loc=len(d1.columns), column=f'Regression Area {i}', value=predict_water4)
#     d1.insert(loc=len(d1.columns), column=f'ANN Area {i}', value=predict_water5)
# d1 = pd.DataFrame(year_new, columns=['Year'])

# for i in range(1, areas+1):
#     gdp_new = np.hstack((gdp.iloc[:, 1], projection(gdp.iloc[:, 1], year, t)))
#     pop_new = np.hstack((population.iloc[:, i], projection(population.iloc[:, i], year, t)))
#     predict_water1 = AGMC_12a(water_consumption.iloc[:, i], pop_new, opt_param_pop[i-1])
#     predict_water2 = AGMC_12a(water_consumption.iloc[:, i], gdp_new, 0.001)
#     predict_water3 = AGMC_12a(water_consumption.iloc[:, i], cost_new, 0.001)
#     predict_water5 = AGMC_13(water_consumption.iloc[:, i], gdp_new, pop_new, 0.0069)
#     predict_water6 = AGMC_13(water_consumption.iloc[:, i], pop_new, cost_new, 0.0069)
#     p1 = linear_param(water_consumption.iloc[:, i], gdp.iloc[:, 1], population.iloc[:, i])
#     predict_water7 = linear_model(gdp_new, pop_new, p1)
#     p2 = linear_param(water_consumption.iloc[:, i], population.iloc[:, i], gdp.iloc[:, 3])
#     predict_water8 = linear_model(pop_new, cost_new, p2)
#     # predict_water9 = ANN(water_consumption.iloc[:, i], gdp_new, pop_new, 200)
#     # predict_water10 = ANN(water_consumption.iloc[:, i], pop_new, cost_new, 200)
#     # Creating a DataFrame
#     d1.insert(loc=len(d1.columns), column=f'AGMC12(Pop) Area {i}', value=predict_water1)
#     d1.insert(loc=len(d1.columns), column=f'AGMC12(GDP) Area {i}', value=predict_water2)
#     d1.insert(loc=len(d1.columns), column=f'AGMC12(Cost) Area {i}', value=predict_water3)
#     d1.insert(loc=len(d1.columns), column=f'AGMC13(GDP) Area {i}', value=predict_water5)
#     d1.insert(loc=len(d1.columns), column=f'AGMC13(Cost) Area {i}', value=predict_water6)
#     d1.insert(loc=len(d1.columns), column=f'Regression(GDP) Area {i}', value=predict_water7)
#     d1.insert(loc=len(d1.columns), column=f'Regression(Cost) Area {i}', value=predict_water8)
#     # d1.insert(loc=len(d1.columns), column=f'ANN(GDP) Area {i}', value=predict_water9)
#     # d1.insert(loc=len(d1.columns), column=f'ANN(Cost) Area {i}', value=predict_water10)
# d1.to_excel('Demand Prediction Results1.xlsx')
